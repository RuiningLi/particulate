from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time

import json
from os.path import join
from typing import List

from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from plyfile import PlyData
import open3d as o3d

from scipy.spatial import cKDTree
from collections import Counter
from partfield.utils import *

#### Export to file #####
def export_colored_mesh_ply(V, F, FL, filename='segmented_mesh.ply'):
    """
    Export a mesh with per-face segmentation labels into a colored PLY file.

    Parameters:
    - V (np.ndarray): Vertices array of shape (N, 3)
    - F (np.ndarray): Faces array of shape (M, 3)
    - FL (np.ndarray): Face labels of shape (M,)
    - filename (str): Output filename
    """
    assert V.shape[1] == 3
    assert F.shape[1] == 3
    assert F.shape[0] == FL.shape[0]

    # Generate distinct colors for each unique label
    unique_labels = np.unique(FL)
    colormap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_color = {
        label: (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
        for i, label in enumerate(unique_labels)
    }

    mesh = trimesh.Trimesh(vertices=V, faces=F)
    FL = np.squeeze(FL)
    for i, face in enumerate(F):
        label = FL[i]
        color = label_to_color[label]
        color_with_alpha = np.append(color, 255)  # Add alpha value
        mesh.visual.face_colors[i] = color_with_alpha

    mesh.export(filename)
    print(f"Exported mesh to {filename}")

def export_pointcloud_with_labels_to_ply(V, VL, filename='colored_pointcloud.ply'):
    """
    Export a labeled point cloud to a PLY file with vertex colors.
    
    Parameters:
    - V: (N, 3) numpy array of XYZ coordinates
    - VL: (N,) numpy array of integer labels
    - filename: Output PLY file name
    """
    assert V.shape[0] == VL.shape[0], "Number of vertices and labels must match"

    # Generate unique colors for each label
    unique_labels = np.unique(VL)
    colormap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_color = {
        label: colormap(i)[:3] for i, label in enumerate(unique_labels)
    }

    VL = np.squeeze(VL)
    # Map labels to RGB colors
    colors = np.array([label_to_color[label] for label in VL])
    
    # Open3D requires colors in float [0, 1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save to .ply
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")
#########################

def construct_face_adjacency_matrix(face_list):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).
    Two faces are adjacent if they share an edge.

    If multiple connected components exist, dummy edges are added to 
    turn them into a single connected component.

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces), 
        containing 1s for adjacent faces and 0s otherwise. 
        Additional edges are added if the faces are in multiple components.
    """

    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    # Step 1: Map each undirected edge -> list of face indices that contain that edge
    edge_to_faces = defaultdict(list)

    # Populate the edge_to_faces dictionary
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # For an edge, we always store its endpoints in sorted order
        # to avoid duplication (e.g. edge (2,5) is the same as (5,2)).
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    # Step 2: Build the adjacency (row, col) lists among faces
    row = []
    col = []
    for e, faces_sharing_e in edge_to_faces.items():
        # If an edge is shared by multiple faces, make each pair of those faces adjacent
        f_indices = list(set(faces_sharing_e))  # unique face indices for this edge
        if len(f_indices) > 1:
            # For each pair of faces, mark them as adjacent
            for i in range(len(f_indices)):
                for j in range(i + 1, len(f_indices)):
                    f_i = f_indices[i]
                    f_j = f_indices[j]
                    row.append(f_i)
                    col.append(f_j)
                    row.append(f_j)
                    col.append(f_i)

    # Create a COO matrix, then convert it to CSR
    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)),
        shape=(num_faces, num_faces)
    ).tocsr()

    return face_adjacency


def relabel_coarse_mesh(dense_mesh, dense_labels, coarse_mesh):
    """
    Relabels a coarse mesh using voting from a dense mesh, where every dense face gets to vote.

    Parameters:
        dense_mesh (trimesh.Trimesh): High-resolution input mesh.
        dense_labels (numpy.ndarray): Per-face labels for the dense mesh (shape: (N_dense_faces,)).
        coarse_mesh (trimesh.Trimesh): Coarser mesh to be relabeled.

    Returns:
        numpy.ndarray: New labels for the coarse mesh (shape: (N_coarse_faces,)).
    """
    # Compute centroids for both dense and coarse mesh faces
    dense_centroids = dense_mesh.vertices[dense_mesh.faces].mean(axis=1)  # (N_dense_faces, 3)
    coarse_centroids = coarse_mesh.vertices[coarse_mesh.faces].mean(axis=1)  # (N_coarse_faces, 3)

    # Use KDTree to efficiently find nearest coarse face for each dense face
    tree = cKDTree(coarse_centroids)
    _, nearest_coarse_faces = tree.query(dense_centroids)  # (N_dense_faces,)

    # Initialize label votes per coarse face
    face_label_votes = {i: [] for i in range(len(coarse_mesh.faces))}

    # Every dense face votes for its nearest coarse face
    dense_labels += 1
    for dense_face_idx, coarse_face_idx in enumerate(nearest_coarse_faces):
        face_label_votes[coarse_face_idx].append(dense_labels[dense_face_idx])

    # Assign new labels based on majority voting
    coarse_labels = np.zeros(len(coarse_mesh.faces), dtype=np.int32)

    for i, votes in face_label_votes.items():
        if votes:  # If this coarse face received votes
            most_common_label = Counter(votes).most_common(1)[0][0]
            coarse_labels[i] = most_common_label
        else:
            coarse_labels[i] = 0  # Mark as unassigned (optional)

    return coarse_labels

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def hierarchical_clustering_labels(children, n_samples, max_cluster=20):
    # Union-Find structure to maintain cluster merges
    uf = UnionFind(2 * n_samples - 1)  # We may need to store up to 2*n_samples - 1 clusters
    
    current_cluster_count = n_samples
    
    # Process merges from the children array
    hierarchical_labels = []
    for i, (child1, child2) in enumerate(children):
        uf.union(child1, i + n_samples)
        uf.union(child2, i + n_samples)
        #uf.union(child1, child2)
        current_cluster_count -= 1  # After each merge, we reduce the cluster count
        
        if current_cluster_count <= max_cluster:
            labels = [uf.find(i) for i in range(n_samples)]
            hierarchical_labels.append(labels)
    
    return hierarchical_labels

def load_ply_to_numpy(filename):
    """
    Load a PLY file and extract the point cloud as a (N, 3) NumPy array.

    Parameters:
        filename (str): Path to the PLY file.

    Returns:
        numpy.ndarray: Point cloud array of shape (N, 3).
    """
    # Read PLY file
    ply_data = PlyData.read(filename)
    
    # Extract vertex data
    vertex_data = ply_data["vertex"]
    
    # Convert to NumPy array (x, y, z)
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    
    return points

def solve_clustering(input_fname, uid, view_id, save_dir="test_results1", max_cluster=20, out_render_fol= "test_render_clustering", filehandle=None, use_agglo=False, max_num_clusters=18, viz_dense=False, export_mesh=True):
    print(uid, view_id)

    try:
        mesh_fname = f'{save_dir}/feat_pca_{uid}_{view_id}.ply'
        dense_mesh = load_mesh_util(mesh_fname)
    except:
        mesh_fname = f'{save_dir}/feat_pca_{uid}_{view_id}_batch.ply'
        dense_mesh = load_mesh_util(mesh_fname)

    vertices = dense_mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.9 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    dense_mesh.vertices = vertices

    ### Load coarse mesh
    input_fname = f'{save_dir}/input_{uid}_{view_id}.ply'
    coarse_mesh = trimesh.load(input_fname, force='mesh')
    vertices = coarse_mesh.vertices

    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * 0.9 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    coarse_mesh.vertices = vertices
    #####################        

    try:
        point_feat = np.load(f'{save_dir}/part_feat_{uid}_{view_id}.npy')
    except:
        try:
            point_feat = np.load(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy')

        except:
            print()
            print("pointfeat loading error. skipping...")
            print(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy')
            return

    point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

    if not use_agglo:
        for num_cluster in range(2, max_num_clusters):
            clustering = KMeans(n_clusters=num_cluster, random_state=0).fit(point_feat)
            labels = clustering.labels_

            if not viz_dense:
                #### Relabel coarse from dense ####
                labels = relabel_coarse_mesh(dense_mesh, labels, coarse_mesh)
                V = coarse_mesh.vertices
                F = coarse_mesh.faces
                ###################################
            else:
                V = dense_mesh.vertices
                F = dense_mesh.faces   

            pred_labels = np.zeros((len(labels), 1))
            for i, label in enumerate(np.unique(labels)):
                # print(i, label)
                pred_labels[labels == label] = i  # Assign RGB values to each label


            fname = str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2)
            fname_clustering = os.path.join(out_render_fol, "cluster_out", str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2))
            np.save(fname_clustering, pred_labels)

            if export_mesh :
                fname_mesh = os.path.join(out_render_fol, "ply", str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2) + ".ply")
                export_colored_mesh_ply(V, F, pred_labels, filename=fname_mesh)  

    else:

        adj_matrix = construct_face_adjacency_matrix(dense_mesh.faces)
        clustering = AgglomerativeClustering(connectivity=adj_matrix,
                                    n_clusters=1,
                                    ).fit(point_feat)
        hierarchical_labels = hierarchical_clustering_labels(clustering.children_, point_feat.shape[0], max_cluster=max_num_clusters)

        all_FL = []
        for n_cluster in range(max_num_clusters):
            print("Processing cluster: "+str(n_cluster))
            labels = hierarchical_labels[n_cluster]
            all_FL.append(labels)
        
        
        all_FL = np.array(all_FL)
        unique_labels = np.unique(all_FL)

        for n_cluster in range(max_num_clusters):
            FL = all_FL[n_cluster]

            if not viz_dense:
                #### Relabel coarse from dense ####
                FL = relabel_coarse_mesh(dense_mesh, FL, coarse_mesh)
                V = coarse_mesh.vertices
                F = coarse_mesh.faces
                ###################################
            else:
                V = dense_mesh.vertices
                F = dense_mesh.faces                

            unique_labels = np.unique(FL)
            relabel = np.zeros((len(FL), 1))
            for i, label in enumerate(unique_labels):
                relabel[FL == label] = i  # Assign RGB values to each label

            if export_mesh :
                fname_mesh = os.path.join(out_render_fol, "ply", str(uid) + "_" + str(view_id) + "_" + str(max_cluster - n_cluster).zfill(2) + ".ply")
                export_colored_mesh_ply(V, F, FL, filename=fname_mesh)        
        
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default= "", type=str)
    parser.add_argument('--root', default= "", type=str)
    parser.add_argument('--dump_dir', default= "", type=str)
    
    parser.add_argument('--max_num_clusters', default= 18, type=int)
    parser.add_argument('--use_agglo', default= True, type=bool)
    parser.add_argument('--viz_dense', default= False, type=bool)
    parser.add_argument('--export_mesh', default= True, type=bool)


    FLAGS = parser.parse_args()
    root = FLAGS.root
    OUTPUT_FOL = FLAGS.dump_dir
    SOURCE_DIR = FLAGS.source_dir

    MAX_NUM_CLUSTERS = FLAGS.max_num_clusters
    USE_AGGLO = FLAGS.use_agglo
    EXPORT_MESH = FLAGS.export_mesh

    models = os.listdir(root)
    os.makedirs(OUTPUT_FOL, exist_ok=True)

    if EXPORT_MESH:
        ply_fol = os.path.join(OUTPUT_FOL, "ply")
        os.makedirs(ply_fol, exist_ok=True)    

    cluster_fol = os.path.join(OUTPUT_FOL, "cluster_out")
    os.makedirs(cluster_fol, exist_ok=True) 

    #### Get existing model_ids ###
    all_files = os.listdir(os.path.join(OUTPUT_FOL, "ply"))

    existing_model_ids = []
    for sample in all_files:
        uid = sample.split("_")[0]
        view_id = sample.split("_")[1]
        # sample_name = str(uid) + "_" + str(view_id)
        sample_name = str(uid)

        if sample_name not in existing_model_ids:
            existing_model_ids.append(sample_name)
    ##############################

    all_files = os.listdir(SOURCE_DIR)
    selected = []
    for f in all_files:
        if (".obj" in f or ".glb" in f) and f.split(".")[0] not in existing_model_ids:
            selected.append(f)
    
    print("Number of models to process: " + str(len(selected)))
    

    for model in selected:
        fname = os.path.join(SOURCE_DIR, model)
        uid = model.split(".")[-2]
        view_id = 0

        solve_clustering(fname, uid, view_id, save_dir=root, out_render_fol= OUTPUT_FOL, use_agglo=USE_AGGLO, max_num_clusters=MAX_NUM_CLUSTERS, viz_dense=FLAGS.viz_dense, export_mesh=EXPORT_MESH)