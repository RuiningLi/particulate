import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim
import potpourri3d as pp3d 
import trimesh
import igl
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from arrgh import arrgh

### For clustering
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from scipy.optimize import linear_sum_assignment

import os, sys
sys.path.append("..")
from partfield.utils import *

@dataclass
class Options:
    
    """ Basic Options """
    filename: str
    filename_alt: str = None

    """System Options"""
    device: str = "cuda"  #  Device
    debug: bool = False  #  enable debug checks
    extras: bool = False # include extra output for viz/debugging

    """ State """
    mode: str = 'co-segmentation'
    m: dict = None          # mesh
    m_alt: dict = None      # second mesh

    # pca mode

    # feature explore mode
    i_feature: int = 0

    i_cluster: int = 1 
    i_cluster2: int = 1 

    i_eps: int = 0.6 

    ### For mixing in clustering
    weight_dist = 1.0
    weight_feat = 1.0
    
    ### For clustering visualization
    independent: bool = True
    source_init: bool = True

    feature_range: float = 0.1
    continuous_explore: bool = False

    viz_mode: str = "faces"

    output_fol: str = "results_pair"

    ### counter for screenshot
    counter: int = 0

modes_list = ['feature_explore', "co-segmentation"]

def load_features(feature_filename, mesh_filename, viz_mode):
    
    print("Reading features:")
    print(f"  Feature filename: {feature_filename}")
    print(f"  Mesh filename: {mesh_filename}")

    # load features
    feat = np.load(feature_filename, allow_pickle=True)
    feat = feat.astype(np.float32)

    # load mesh things
    tm =  load_mesh_util(mesh_filename)

    V = np.array(tm.vertices, dtype=np.float32)
    F = np.array(tm.faces)

    if viz_mode ==  "faces":
        pca_colors = np.array(tm.visual.face_colors, dtype=np.float32)
        pca_colors = pca_colors[:,:3] / 255.
        
    else:
        pca_colors = np.array(tm.visual.vertex_colors, dtype=np.float32)
        pca_colors = pca_colors[:,:3] / 255.

    arrgh(V, F, pca_colors, feat)

    return {
        'V' : V, 
        'F' : F, 
        'pca_colors' : pca_colors, 
        'feat_np' : feat,
        'feat_pt' : torch.tensor(feat, device='cuda'),
        'trimesh' : tm,
        'label' : None,
        'num_cluster' : 1,
        'scalar' : None
    }

def prep_feature_mesh(m, name='mesh'):
    ps_mesh = ps.register_surface_mesh(name, m['V'], m['F'])
    ps_mesh.set_selection_mode('faces_only')
    m['ps_mesh'] = ps_mesh

def viz_pca_colors(m):
    m['ps_mesh'].add_color_quantity('pca colors', m['pca_colors'], enabled=True, defined_on=m["viz_mode"])

def viz_feature(m, ind):
    m['ps_mesh'].add_scalar_quantity('pca colors', m['feat_np'][:,ind], cmap='turbo', enabled=True, defined_on=m["viz_mode"])

def feature_distance_np(feats, query_feat):
    # normalize
    feats = feats / np.linalg.norm(feats,axis=1)[:,None]
    query_feat = query_feat / np.linalg.norm(query_feat)
    # cosine distance
    cos_sim = np.dot(feats, query_feat)
    cos_dist = (1 - cos_sim) / 2.
    return cos_dist

def feature_distance_pt(feats, query_feat):
    return (1. - torch.nn.functional.cosine_similarity(feats, query_feat[None,:], dim=-1)) / 2.


def ps_callback(opts):
    m = opts.m

    changed, ind = psim.Combo("Mode", modes_list.index(opts.mode), modes_list)
    if changed:
        opts.mode = modes_list[ind]
        m['ps_mesh'].remove_all_quantities()
        if opts.m_alt is not None:
            opts.m_alt['ps_mesh'].remove_all_quantities()
    
    elif opts.mode == 'feature_explore':
        psim.TextUnformatted("Click on the mesh on the left")
        psim.TextUnformatted("to highlight all faces within a given radius in feature space.""")

        io = psim.GetIO()
        if io.MouseClicked[0] or opts.continuous_explore:
            screen_coords = io.MousePos
            cam_params = ps.get_view_camera_parameters()

            pick_result = ps.pick(screen_coords=screen_coords)

            # Check if we hit one of the meshes
            if pick_result.is_hit and pick_result.structure_name == "mesh":
                if pick_result.structure_data['element_type'] != "face":
                    # shouldn't be possible
                    raise ValueError("pick returned non-face")

                f_hit = pick_result.structure_data['index']
                bary_weights = np.array(pick_result.structure_data['bary_coords'])

                # get the feature via interpolation
                point_feat = m['feat_np'][f_hit,:]
                point_feat_pt = torch.tensor(point_feat, device='cuda')

                all_dists1 = feature_distance_pt(m['feat_pt'], point_feat_pt).detach().cpu().numpy()
                m['ps_mesh'].add_scalar_quantity("distance", all_dists1, cmap='blues', vminmax=(0, opts.feature_range), enabled=True, defined_on=m["viz_mode"])
                opts.m['scalar'] = all_dists1

                if opts.m_alt is not None:
                    all_dists2 = feature_distance_pt(opts.m_alt['feat_pt'], point_feat_pt).detach().cpu().numpy()
                    opts.m_alt['ps_mesh'].add_scalar_quantity("distance", all_dists2, cmap='blues', vminmax=(0, opts.feature_range), enabled=True, defined_on=m["viz_mode"])
                    opts.m_alt['scalar'] = all_dists2

            else:
                # not hit
                pass

        if psim.Button("Export"):
            ### Save output
            OUTPUT_FOL = opts.output_fol
            fname1 = opts.filename
            out_mesh_file = os.path.join(OUTPUT_FOL, fname1+'.obj')

            igl.write_obj(out_mesh_file, opts.m["V"], opts.m["F"])
            print("Saved '{}'.".format(out_mesh_file))

            out_face_ids_file = os.path.join(OUTPUT_FOL, fname1 + '_feat_dist_' + str(opts.counter) +'.txt')
            np.savetxt(out_face_ids_file, opts.m['scalar'], fmt='%f')
            print("Saved '{}'.".format(out_face_ids_file))


            fname2 = opts.filename_alt
            out_mesh_file = os.path.join(OUTPUT_FOL, fname2+'.obj')

            igl.write_obj(out_mesh_file,  opts.m_alt["V"], opts.m_alt["F"])
            print("Saved '{}'.".format(out_mesh_file))

            out_face_ids_file = os.path.join(OUTPUT_FOL, fname2 + '_feat_dist_' + str(opts.counter) +'.txt')
            np.savetxt(out_face_ids_file, opts.m_alt['scalar'], fmt='%f')
            # print("Saved '{}'.".format(out_face_ids_file))

            opts.counter += 1  


        _, opts.feature_range = psim.SliderFloat('range', opts.feature_range, v_min=0., v_max=1.0, power=3)
        _, opts.continuous_explore = psim.Checkbox('continuous', opts.continuous_explore)

        # TODO nsharp remember how the keycodes work
        if io.KeysDown[ord('q')]:
            opts.feature_range += 0.01
        if io.KeysDown[ord('w')]:
            opts.feature_range -= 0.01
    

    elif opts.mode == "co-segmentation":

        changed, opts.source_init = psim.Checkbox("Source Init", opts.source_init)
        changed, opts.independent = psim.Checkbox("Independent", opts.independent)

        psim.TextUnformatted("Use the slider to toggle the number of desired clusters.")
        cluster_changed, opts.i_cluster = psim.SliderInt("num clusters for model1", opts.i_cluster, v_min=1, v_max=30)
        cluster_changed, opts.i_cluster2 = psim.SliderInt("num clusters for model2", opts.i_cluster2, v_min=1, v_max=30)

        # if cluster_changed:
        if psim.Button("Recompute"):

            ### Run clustering algorithm

            ### Mesh 1
            num_clusters1 = opts.i_cluster
            point_feat1 = m['feat_np']
            point_feat1 = point_feat1 / np.linalg.norm(point_feat1, axis=-1, keepdims=True)
            clustering1 = KMeans(n_clusters=num_clusters1, random_state=0, n_init="auto").fit(point_feat1)

            ### Get feature means per cluster
            feature_means1 = []
            for j in range(num_clusters1):
                all_cluster_feat = point_feat1[clustering1.labels_==j]
                mean_feat = np.mean(all_cluster_feat, axis=0)
                feature_means1.append(mean_feat)

            feature_means1 = np.array(feature_means1)
            tree = KDTree(feature_means1)


            if opts.source_init:
                num_clusters2 = opts.i_cluster
                init_mode = np.array(feature_means1)

            ## default is kmeans++
            else:
                num_clusters2 = opts.i_cluster2
                init_mode = "k-means++"

            ### Mesh 2
            point_feat2 = opts.m_alt['feat_np']
            point_feat2 = point_feat2 / np.linalg.norm(point_feat2, axis=-1, keepdims=True)

            clustering2 = KMeans(n_clusters=num_clusters2, random_state=0, init=init_mode).fit(point_feat2)

            ### Get feature means per cluster
            feature_means2 = []
            for j in range(num_clusters2):
                all_cluster_feat = point_feat2[clustering2.labels_==j]
                mean_feat = np.mean(all_cluster_feat, axis=0)
                feature_means2.append(mean_feat)

            feature_means2 = np.array(feature_means2)
            _, nn_idx = tree.query(feature_means2, k=1)

            print(nn_idx)
            print("Both KMeans")
            print(np.unique(clustering1.labels_))
            print(np.unique(clustering2.labels_))

            relabelled_2 = nn_idx[clustering2.labels_]

            print(np.unique(relabelled_2))
            print()

            m['ps_mesh'].add_scalar_quantity("cluster_both_kmeans", clustering1.labels_, cmap='turbo', vminmax=(0, num_clusters1-1), enabled=True, defined_on=m["viz_mode"])
            opts.m['label'] = clustering1.labels_
            opts.m['num_cluster'] = num_clusters1

            if opts.independent:
                opts.m_alt['ps_mesh'].add_scalar_quantity("cluster", clustering2.labels_, cmap='turbo', vminmax=(0, num_clusters2-1), enabled=True, defined_on=m["viz_mode"])
                opts.m_alt['label'] = clustering2.labels_
                opts.m_alt['num_cluster'] = num_clusters2
            else:
                opts.m_alt['ps_mesh'].add_scalar_quantity("cluster", relabelled_2, cmap='turbo', vminmax=(0, num_clusters1-1), enabled=True, defined_on=m["viz_mode"])
                opts.m_alt['label'] = relabelled_2
                opts.m_alt['num_cluster'] = num_clusters1


        if psim.Button("Export"):
            ### Save output
            OUTPUT_FOL = opts.output_fol
            fname1 = opts.filename
            out_mesh_file = os.path.join(OUTPUT_FOL, fname1+'.obj')

            igl.write_obj(out_mesh_file, opts.m["V"], opts.m["F"])
            print("Saved '{}'.".format(out_mesh_file))

            if m["viz_mode"] == "faces":
                out_face_ids_file = os.path.join(OUTPUT_FOL, fname1 + "_" + str(opts.m['num_cluster']) + '_pred_face_ids.txt')
            else:
                out_face_ids_file = os.path.join(OUTPUT_FOL, fname1 + "_" + str(opts.m['num_cluster']) + '_pred_vertices_ids.txt')

            np.savetxt(out_face_ids_file, opts.m['label'], fmt='%d')
            print("Saved '{}'.".format(out_face_ids_file))


            fname2 = opts.filename_alt
            out_mesh_file = os.path.join(OUTPUT_FOL, fname2 +'.obj')

            igl.write_obj(out_mesh_file,  opts.m_alt["V"], opts.m_alt["F"])
            print("Saved '{}'.".format(out_mesh_file))

            if m["viz_mode"] == "faces":
                out_face_ids_file = os.path.join(OUTPUT_FOL, fname2 + "_" + str(opts.m_alt['num_cluster']) + '_pred_face_ids.txt')
            else:
                out_face_ids_file = os.path.join(OUTPUT_FOL, fname2 + "_" + str(opts.m_alt['num_cluster']) + '_pred_vertices_ids.txt')

            np.savetxt(out_face_ids_file, opts.m_alt['label'], fmt='%d')
            print("Saved '{}'.".format(out_face_ids_file))


def main():
    ## Parse args
    # Uses simple_parsing library to automatically construct parser from the dataclass Options
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    parser.add_argument('--data_root', default="../exp_results/partfield_features/trellis", help='Path the model features are stored.')
    args = parser.parse_args()
    opts: Options = args.options

    DATA_ROOT = args.data_root

    shape_1 = opts.filename
    shape_2 = opts.filename_alt

    if os.path.exists(os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0.npy")):
        feature_fname1 = os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0.npy")
        feature_fname2 = os.path.join(DATA_ROOT, "part_feat_"+ shape_2 + "_0.npy")

        mesh_fname1 = os.path.join(DATA_ROOT, "feat_pca_"+ shape_1 + "_0.ply")
        mesh_fname2 = os.path.join(DATA_ROOT, "feat_pca_"+ shape_2 + "_0.ply")
    else:
        feature_fname1 = os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0_batch.npy")
        feature_fname2 = os.path.join(DATA_ROOT, "part_feat_"+ shape_2 + "_0_batch.npy")

        mesh_fname1 = os.path.join(DATA_ROOT, "feat_pca_"+ shape_1 + "_0.ply")
        mesh_fname2 = os.path.join(DATA_ROOT, "feat_pca_"+ shape_2 + "_0.ply")

    #### To save output ####
    os.makedirs(opts.output_fol, exist_ok=True)      
    ########################

    # Initialize
    ps.init()

    mesh_dict = load_features(feature_fname1, mesh_fname1, opts.viz_mode)
    prep_feature_mesh(mesh_dict)
    mesh_dict["viz_mode"] = opts.viz_mode
    opts.m = mesh_dict

    mesh_dict_alt = load_features(feature_fname2, mesh_fname2, opts.viz_mode)
    prep_feature_mesh(mesh_dict_alt, name='mesh_alt')
    mesh_dict_alt['ps_mesh'].translate((2.5, 0., 0.))
    mesh_dict_alt["viz_mode"] = opts.viz_mode
    opts.m_alt = mesh_dict_alt

    # Start the interactive UI
    ps.set_user_callback(lambda : ps_callback(opts))
    ps.show()


if __name__ == "__main__":
    main()

