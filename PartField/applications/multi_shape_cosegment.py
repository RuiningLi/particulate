import numpy as np
import torch
import argparse
from dataclasses import dataclass

from arrgh import arrgh
import polyscope as ps
import polyscope.imgui as psim
import potpourri3d as pp3d 
import trimesh

import cuml
import xgboost as xgb

import os, random

import sys
sys.path.append("..")
from partfield.utils import *

@dataclass
class State:

    objects = None
    train_objects = None
    
    # Input options
    subsample_inputs: int = -1
    n_train_subset: int = 0

    # Label
    N_class: int = 2

    # Annotations
    # A annotations (initially A = 0)
    anno_feat: np.array = np.zeros((0,448), dtype=np.float32) # [A,F]
    anno_label: np.array = np.zeros((0,), dtype=np.int32) # [A]
    anno_pos: np.array = np.zeros((0,3), dtype=np.float32) # [A,3]

    # Intermediate selection data
    is_selecting: bool = False
    selection_class: int = 0
    
    # Fitting algorithm
    fit_to: str = "Annotations"
    fit_method : str = "LogisticRegression"
    auto_update_fit: bool = True
    
    # Training data
    # T training datapoints
    train_feat: np.array = np.zeros((0,448), dtype=np.float32) # [T,F]
    train_label: np.array = np.zeros((0,), dtype=np.int32) # [T]

    # Viz
    grid_w : int = 8
    per_obj_shift : float = 2.
    anno_radius : float = 0.01
    ps_cloud_annotation = None
    ps_structure_name_to_index_map = {}


fit_methods_list = ["LinearRegression", "LogisticRegression", "LinearSVC", "RandomForest", "NearestNeighbors", "XGBoost"]
fit_to_list = ["Annotations", "TrainingSet"]

def load_mesh_and_features(mesh_filepath, ind, require_gt=False, gt_label_fol = ""):

    dirpath, filename = os.path.split(mesh_filepath)
    filename_core = filename[9:-6] # splits off "feat_pca_" ... "_0.ply"
    feature_filename = "part_feat_"+ filename_core + "_0_batch.npy"
    feature_filepath = os.path.join(dirpath, feature_filename)
    
    gt_filename = filename_core + ".seg"
    gt_filepath = os.path.join(gt_label_fol, gt_filename)
    have_gt = os.path.isfile(gt_filepath)

    print("  Reading file:")
    print(f"    Mesh filename: {mesh_filepath}")
    print(f"    Feature filename: {feature_filepath}")
    print(f"    Ground Truth Label filename: {gt_filepath}  --  present = {have_gt}")

    # load features
    feat = np.load(feature_filepath, allow_pickle=False)
    feat = feat.astype(np.float32)

    # load mesh things
    # TODO replace this with just loading V/F from numpy archive
    tm =  load_mesh_util(mesh_filepath)

    V = np.array(tm.vertices, dtype=np.float32)
    F = np.array(tm.faces)

    # load ground truth, if available
    if have_gt:
        gt_labels = np.loadtxt(gt_filepath)
        gt_labels = gt_labels.astype(np.int32) - 1
    else:
        if require_gt:
            raise ValueError("could not find ground-truth file, but it is required")
        gt_labels = None

    # pca_colors = None

    return {
        'nicename' : f"{ind:02d}_{filename_core}",
        'mesh_filepath' : mesh_filepath,
        'feature_filepath' : feature_filepath,
        'V' : V, 
        'F' : F, 
        'feat_np' : feat,
        # 'feat_pt' : torch.tensor(feat, device='cuda'),
        'gt_labels' : gt_labels
    }

def shift_for_ind(state : State, ind):

    x_ind = ind % state.grid_w
    y_ind = ind // state.grid_w

    shift = np.array([state.per_obj_shift * x_ind, 0, -state.per_obj_shift * y_ind])

    return shift

def viz_upper_limit(state : State, ind_count):

    x_max = min(ind_count, state.grid_w)
    y_max = ind_count // state.grid_w

    bound = np.array([state.per_obj_shift * x_max, 0, -state.per_obj_shift * y_max])

    return bound


def initialize_object_viz(state : State, obj, index=0):
    obj['ps_mesh'] = ps.register_surface_mesh(obj['nicename'], obj['V'], obj['F'], color=(.8, .8, .8))
    shift = shift_for_ind(state, index)
    obj['ps_mesh'].translate(shift)
    obj['ps_mesh'].set_selection_mode('faces_only')
    state.ps_structure_name_to_index_map[obj['nicename']] = index

def update_prediction(state: State):

    print("Updating predictions..")

    N_anno = state.anno_label.shape[0]

    # Quick out if we don't have at least two distinct class labels present
    if(state.fit_to == "Annotations" and len(np.unique(state.anno_label)) <= 1):
        return state
    
    # Quick out if we don't have 
    if(state.fit_to == "TrainingSet" and state.train_objects is None):
        return state
    
    if state.fit_method == "LinearRegression":
        classifier = cuml.multiclass.MulticlassClassifier(cuml.linear_model.LinearRegression(), strategy='ovr')
    elif state.fit_method == "LogisticRegression":
        classifier = cuml.multiclass.MulticlassClassifier(cuml.linear_model.LogisticRegression(), strategy='ovr')
    elif state.fit_method == "LinearSVC":
        classifier = cuml.multiclass.MulticlassClassifier(cuml.svm.LinearSVC(), strategy='ovr')
    elif state.fit_method == "RandomForest":
        classifier = cuml.ensemble.RandomForestClassifier()
    elif state.fit_method == "NearestNeighbors":
        classifier = cuml.multiclass.MulticlassClassifier(cuml.neighbors.KNeighborsRegressor(n_neighbors=1), strategy='ovr')
    elif state.fit_method == "XGBoost":
        classifier = xgb.XGBClassifier(max_depth=7, n_estimators=1000)
    else:
        raise ValueError("unrecognized fit method")

    if state.fit_to == "TrainingSet":

        all_train_feats = []
        all_train_labels = []
        for obj in state.train_objects:
            all_train_feats.append(obj['feat_np'])
            all_train_labels.append(obj['gt_labels'])

        all_train_feats = np.concatenate(all_train_feats, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)

        state.N_class = np.max(all_train_labels) + 1
        
        classifier.fit(all_train_feats, all_train_labels)


    elif state.fit_to == "Annotations":
        classifier.fit(state.anno_feat,state.anno_label)
    else:
        raise ValueError("unrecognized fit to")

    n_total = 0
    n_correct = 0

    for obj in state.objects:
        obj['pred_label'] = classifier.predict(obj['feat_np'])

        if obj['gt_labels'] is not None:
            n_total += obj['gt_labels'].shape[0]
            n_correct += np.sum(obj['pred_label'] == obj['gt_labels'], dtype=np.int32)

    if(state.fit_to == "TrainingSet" and n_total > 0):
        frac = n_correct / n_total
        print(f"Test accuracy: {n_correct:d} / {n_total:d}   {100*frac:.02f}%")
    
    
    print("Done updating predictions.")

    return state

def update_prediction_viz(state: State):

    for obj in state.objects:
        if 'pred_label' in obj:
            obj['ps_mesh'].add_scalar_quantity("pred labels", obj['pred_label'], defined_on='faces', vminmax=(0,state.N_class-1), cmap='turbo', enabled=True)

    return state

def update_annotation_viz(state: State):

    ps_cloud = ps.register_point_cloud("annotations", state.anno_pos, radius=state.anno_radius, material='candy')
    ps_cloud.add_scalar_quantity("labels", state.anno_label, vminmax=(0,state.N_class-1), cmap='turbo', enabled=True)

    state.ps_cloud_annotation = ps_cloud

    return state


def filter_old_labels(state: State):
    """
    Filter out annotations from classes that don't exist any more
    """

    keep_mask = state.anno_label < state.N_class
    state.anno_feat = state.anno_feat[keep_mask,:]
    state.anno_label = state.anno_label[keep_mask]
    state.anno_pos = state.anno_pos[keep_mask,:]

    return state

def undo_last_annotation(state: State):

    state.anno_feat = state.anno_feat[:-1,:]
    state.anno_label = state.anno_label[:-1]
    state.anno_pos = state.anno_pos[:-1,:]

    return state

def ps_callback(state_list):
    state : State = state_list[0] # hacky pass-by-reference, since we want to edit it below


    # If we're in selection mode, that's the only thing we can do
    if state.is_selecting:

        psim.TextUnformatted(f"Annotating class {state.selection_class:02d}. Click on any mesh face.")

        io = psim.GetIO()
        if io.MouseClicked[0]:
            screen_coords = io.MousePos
            pick_result = ps.pick(screen_coords=screen_coords)

            # Check if we hit one of the meshes
            if pick_result.is_hit and pick_result.structure_name in state.ps_structure_name_to_index_map:
                if pick_result.structure_data['element_type'] != "face":
                    # shouldn't be possible
                    raise ValueError("pick returned non-face")

                i_obj = state.ps_structure_name_to_index_map[pick_result.structure_name]
                f_hit = pick_result.structure_data['index']

                obj = state.objects[i_obj]
                V = obj['V']
                F = obj['F']
                feat = obj['feat_np']

                face_corners = V[F[f_hit,:],:]
                new_anno_feat = feat[f_hit,:]
                new_anno_label = state.selection_class
                new_anno_pos = np.mean(face_corners, axis=0) + shift_for_ind(state, i_obj)

                state.anno_feat = np.concatenate((state.anno_feat, new_anno_feat[None,:]))
                state.anno_label = np.concatenate((state.anno_label, np.array((new_anno_label,))))
                state.anno_pos = np.concatenate((state.anno_pos, new_anno_pos[None,:]))

                state = update_annotation_viz(state)
                state.is_selecting = False
                needs_pred_update = True

                if state.auto_update_fit:
                    state = update_prediction(state)
                    state = update_prediction_viz(state)


        return
        
    # If not selecting, build the main UI
    needs_pred_update = False
        
    psim.PushItemWidth(150)
    changed, state.N_class = psim.InputInt("N_class", state.N_class, step=1)
    psim.PopItemWidth()
    if changed:
        state = filter_old_labels(state)
        state = update_annotation_viz(state)


    # Check for keypress annotation
    io = psim.GetIO()
    class_keys = { 'w' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9,}
    for c in class_keys:
        if class_keys[c] >= state.N_class: 
            continue

        if psim.IsKeyPressed(ps.get_key_code(c)):
            state.is_selecting = True
            state.selection_class = class_keys[c]


    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if(psim.TreeNode("Annotate")):

        psim.TextUnformatted("New class annotation. Select class to add add annotation for:")
        psim.TextUnformatted("(alternately, press key {w,1,2,3,4...})")
        for i_class in range(state.N_class):
            
            if i_class > 0:
                psim.SameLine()

            if psim.Button(f"{i_class:02d}"):
                state.is_selecting = True
                state.selection_class = i_class
            

        if psim.Button("Undo Last Annotation"):
            state = undo_last_annotation(state)
            state = update_annotation_viz(state)
            needs_pred_update = True
            


        psim.TreePop()

    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if(psim.TreeNode("Fit")):
    
        psim.PushItemWidth(150)
        
        changed, ind = psim.Combo("Fit To", fit_to_list.index(state.fit_to), fit_to_list)
        if changed:
            state.fit_to = fit_methods_list[ind]
            needs_pred_update = True

        changed, ind = psim.Combo("Fit Method", fit_methods_list.index(state.fit_method), fit_methods_list)
        if changed:
            state.fit_method = fit_methods_list[ind]
            needs_pred_update = True

        if psim.Button("Update fit"):
            state = update_prediction(state)
            state = update_prediction_viz(state)

        psim.SameLine()

        changed, state.auto_update_fit = psim.Checkbox("Auto-update fit", state.auto_update_fit)
        if changed:
            needs_pred_update = True
        
    
        psim.PopItemWidth()
    
        psim.TreePop()
    
    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
    if(psim.TreeNode("Visualization")):

        psim.PushItemWidth(150)
        changed, state.anno_radius = psim.SliderFloat("Annotation Point Radius", state.anno_radius, 0.00001, 0.02)
        if changed:
            state = update_annotation_viz(state)
        psim.PopItemWidth()

        psim.TreePop()
        

    if needs_pred_update and state.auto_update_fit:
        state = update_prediction(state)
        state = update_prediction_viz(state)


def main():
    
    state = State()

    ## Parse args
    parser = argparse.ArgumentParser()

    parser.add_argument('--meshes', nargs='+', help='List of meshes to process.', required=True)
    parser.add_argument('--n_train_subset', default=0, help='How many meshes to train on.')
    parser.add_argument('--gt_label_fol', default="../data/coseg_guitar/gt", help='Path where labels are stored.')
    parser.add_argument('--subsample_inputs', default=state.subsample_inputs, help='Only show a random fraction of inputs')
    parser.add_argument('--per_obj_shift', default=state.per_obj_shift, help='How to space out objects in UI grid')
    parser.add_argument('--grid_w', default=state.grid_w, help='Grid width')

    args = parser.parse_args()


    state.n_train_subset = int(args.n_train_subset)
    state.subsample_inputs = int(args.subsample_inputs)
    state.per_obj_shift = float(args.per_obj_shift)
    state.grid_w = int(args.grid_w)

    ## Load data
    # First, resolve directories to load all files in directory
    all_filepaths = []
    print("Resolving passed directories")
    for entry in args.meshes:
        if os.path.isdir(entry):
            dir_path = entry
            print(f"  processing directory {dir_path}")
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path) and file_path.endswith(".ply") and "feat_pca" in file_path:
                    print(f"    adding file {file_path}")
                    all_filepaths.append(file_path)
        else:
            all_filepaths.append(entry)

    random.shuffle(all_filepaths)

    if state.subsample_inputs != -1:
        all_filepaths = all_filepaths[:state.subsample_inputs]


    if state.n_train_subset != 0:

        print(state.n_train_subset)

        train_filepaths = all_filepaths[:state.n_train_subset]
        all_filepaths = all_filepaths[state.n_train_subset:]

        print(f"Loading {len(train_filepaths)} files")
        state.train_objects = []
        for i, file_path in enumerate(train_filepaths):
            state.train_objects.append(load_mesh_and_features(file_path, i, require_gt=True, gt_label_fol=args.gt_label_fol))
        
        state.fit_to = "TrainingSet"

    # Load files
    print(f"Loading {len(all_filepaths)} files")
    state.objects = []
    for i, file_path in enumerate(all_filepaths):
        state.objects.append(load_mesh_and_features(file_path, i))


    ## Set up visualization
    ps.init()
    ps.set_automatically_compute_scene_extents(False)
    lim = viz_upper_limit(state, len(state.objects))
    ps.set_length_scale(np.linalg.norm(lim) / 4.)
    low = np.array((0, -1., -1.)) 
    high = lim
    ps.set_bounding_box(low, high)

    for ind, o in enumerate(state.objects):
        initialize_object_viz(state, o, ind)

    print(f"Loaded {len(state.objects)} objects")
    if state.n_train_subset != 0:
        print(f"Loaded {len(state.train_objects)} training objects")
        
    # One first prediction
    # (does nothing if there is no annotatoins / training data)    
    state = update_prediction(state)
    state = update_prediction_viz(state)

    # Start the interactive UI
    ps.set_user_callback(lambda : ps_callback([state]))
    ps.show()


if __name__ == "__main__":
    main()

