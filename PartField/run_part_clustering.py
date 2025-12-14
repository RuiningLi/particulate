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
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from plyfile import PlyData
import open3d as o3d
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

#########################
def construct_face_adjacency_matrix_ccmst(face_list, vertices, k=10, with_knn=True):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).

    Two faces are adjacent if they share an edge (the "mesh adjacency").
    If multiple connected components remain, we:
      1) Compute the centroid of each connected component as the mean of all face centroids.
      2) Use a KNN graph (k=10) based on centroid distances on each connected component.
      3) Compute MST of that KNN graph.
      4) Add MST edges that connect different components as "dummy" edges
         in the face adjacency matrix, ensuring one connected component. The selected face for 
         each connected component is the face closest to the component centroid.

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.
    vertices : np.ndarray of shape (num_vertices, 3)
        Array of vertex coordinates.
    k : int, optional
        Number of neighbors to use in centroid KNN. Default is 10.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces),
        containing 1s for adjacent faces (shared-edge adjacency)
        plus dummy edges ensuring a single connected component.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    #--------------------------------------------------------------------------
    # 1) Build adjacency based on shared edges.
    #    (Same logic as the original code, plus import statements.)
    #--------------------------------------------------------------------------
    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # Sort each edge’s endpoints so (i, j) == (j, i)
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            # For every pair of distinct faces that share this edge,
            # mark them as mutually adjacent
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    #--------------------------------------------------------------------------
    # 2) Check if the graph from shared edges is already connected.
    #--------------------------------------------------------------------------
    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1
    print("n_components", n_components)

    if n_components == 1:
        # Already a single connected component, no need for dummy edges
        return face_adjacency

    #--------------------------------------------------------------------------
    # 3) Compute centroids of each face for building a KNN graph.
    #--------------------------------------------------------------------------
    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    # #--------------------------------------------------------------------------
    # # 4a) Build a KNN graph (k=10) over face centroids using scikit‐learn
    # #--------------------------------------------------------------------------
    # knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    # knn.fit(face_centroids)
    # distances, indices = knn.kneighbors(face_centroids)
    # # 'distances[i]' are the distances from face i to each of its 'k' neighbors
    # # 'indices[i]' are the face indices of those neighbors

    #--------------------------------------------------------------------------
    # 4b) Build a KNN graph on connected components
    #--------------------------------------------------------------------------
    # Group faces by their root representative in the Union-Find structure
    component_dict = {}
    for face_idx in range(num_faces):
        root = uf.find(face_idx)
        if root not in component_dict:
            component_dict[root] = set()
        component_dict[root].add(face_idx)

    connected_components = list(component_dict.values())
    
    print("Using connected component MST.")
    component_centroid_face_idx = []
    connected_component_centroids = []
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    for component in connected_components:
        curr_component_faces = list(component)
        curr_component_face_centroids = face_centroids[curr_component_faces]
        component_centroid = np.mean(curr_component_face_centroids, axis=0)

        ### Assign a face closest to the centroid
        face_idx = curr_component_faces[np.argmin(np.linalg.norm(curr_component_face_centroids-component_centroid, axis=-1))]

        connected_component_centroids.append(component_centroid)
        component_centroid_face_idx.append(face_idx)

    component_centroid_face_idx = np.array(component_centroid_face_idx)
    connected_component_centroids = np.array(connected_component_centroids)

    if n_components < k:
        knn = NearestNeighbors(n_neighbors=n_components, algorithm='auto')
    else:
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(connected_component_centroids)
    distances, indices = knn.kneighbors(connected_component_centroids)    

    #--------------------------------------------------------------------------
    # 5) Build a weighted graph in NetworkX using centroid-distances as edges
    #--------------------------------------------------------------------------
    G = nx.Graph()
    # Add each face as a node in the graph
    G.add_nodes_from(range(num_faces))

    # For each face i, add edges (i -> j) for each neighbor j in the KNN
    for idx1 in range(n_components):
        i = component_centroid_face_idx[idx1]
        for idx2, dist in zip(indices[idx1], distances[idx1]):
            j = component_centroid_face_idx[idx2]
            if i == j:
                continue  # skip self-loop
            # Add an undirected edge with 'weight' = distance
            # NetworkX handles parallel edges gracefully via last add_edge,
            # but it typically overwrites the weight if (i, j) already exists.
            G.add_edge(i, j, weight=dist)

    #--------------------------------------------------------------------------
    # 6) Compute MST on that KNN graph
    #--------------------------------------------------------------------------
    mst = nx.minimum_spanning_tree(G, weight='weight')
    # Sort MST edges by ascending weight, so we add the shortest edges first
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )
    print("mst edges sorted", len(mst_edges_sorted))
    #--------------------------------------------------------------------------
    # 7) Use a union-find structure to add MST edges only if they
    #    connect two currently disconnected components of the adjacency matrix
    #--------------------------------------------------------------------------

    # Convert face_adjacency to LIL format for efficient edge addition
    adjacency_lil = face_adjacency.tolil()

    # Now, step through MST edges in ascending order
    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            # These belong to different components, so unify them
            uf.union(u, v)
            # And add a "dummy" edge to our adjacency matrix
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    # Convert back to CSR format and return
    face_adjacency = adjacency_lil.tocsr()

    if with_knn:
        print("Adding KNN edges.")
        ### Add KNN edges graph too
        dummy_row = []
        dummy_col = []
        for idx1 in range(n_components):
            i = component_centroid_face_idx[idx1]
            for idx2 in indices[idx1]:
                j = component_centroid_face_idx[idx2]     
                dummy_row.extend([i, j])
                dummy_col.extend([j, i]) ### duplicates are handled by coo

        dummy_data = np.ones(len(dummy_row), dtype=np.int16)
        dummy_mat = coo_matrix(
            (dummy_data, (dummy_row, dummy_col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        face_adjacency = face_adjacency + dummy_mat
        ###########################

    return face_adjacency
#########################

def construct_face_adjacency_matrix_facemst(face_list, vertices, k=10, with_knn=True):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).

    Two faces are adjacent if they share an edge (the "mesh adjacency").
    If multiple connected components remain, we:
      1) Compute the centroid of each face.
      2) Use a KNN graph (k=10) based on centroid distances.
      3) Compute MST of that KNN graph.
      4) Add MST edges that connect different components as "dummy" edges
         in the face adjacency matrix, ensuring one connected component.

    Parameters
    ----------
    face_list : list of tuples
        List of faces, each face is a tuple (v0, v1, v2) of vertex indices.
    vertices : np.ndarray of shape (num_vertices, 3)
        Array of vertex coordinates.
    k : int, optional
        Number of neighbors to use in centroid KNN. Default is 10.

    Returns
    -------
    face_adjacency : scipy.sparse.csr_matrix
        A CSR sparse matrix of shape (num_faces, num_faces),
        containing 1s for adjacent faces (shared-edge adjacency)
        plus dummy edges ensuring a single connected component.
    """
    num_faces = len(face_list)
    if num_faces == 0:
        # Return an empty matrix if no faces
        return csr_matrix((0, 0))

    #--------------------------------------------------------------------------
    # 1) Build adjacency based on shared edges.
    #    (Same logic as the original code, plus import statements.)
    #--------------------------------------------------------------------------
    edge_to_faces = defaultdict(list)
    uf = UnionFind(num_faces)
    for f_idx, (v0, v1, v2) in enumerate(face_list):
        # Sort each edge’s endpoints so (i, j) == (j, i)
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0)))
        ]
        for e in edges:
            edge_to_faces[e].append(f_idx)

    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        unique_faces = list(set(face_indices))
        if len(unique_faces) > 1:
            # For every pair of distinct faces that share this edge,
            # mark them as mutually adjacent
            for i in range(len(unique_faces)):
                for j in range(i + 1, len(unique_faces)):
                    fi = unique_faces[i]
                    fj = unique_faces[j]
                    row.append(fi)
                    col.append(fj)
                    row.append(fj)
                    col.append(fi)
                    uf.union(fi, fj)

    data = np.ones(len(row), dtype=np.int8)
    face_adjacency = coo_matrix(
        (data, (row, col)), shape=(num_faces, num_faces)
    ).tocsr()

    #--------------------------------------------------------------------------
    # 2) Check if the graph from shared edges is already connected.
    #--------------------------------------------------------------------------
    n_components = 0
    for i in range(num_faces):
        if uf.find(i) == i:
            n_components += 1
    print("n_components", n_components)

    if n_components == 1:
        # Already a single connected component, no need for dummy edges
        return face_adjacency
    #--------------------------------------------------------------------------
    # 3) Compute centroids of each face for building a KNN graph.
    #--------------------------------------------------------------------------
    face_centroids = []
    for (v0, v1, v2) in face_list:
        centroid = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        face_centroids.append(centroid)
    face_centroids = np.array(face_centroids)

    #--------------------------------------------------------------------------
    # 4) Build a KNN graph (k=10) over face centroids using scikit‐learn
    #--------------------------------------------------------------------------
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(face_centroids)
    distances, indices = knn.kneighbors(face_centroids)
    # 'distances[i]' are the distances from face i to each of its 'k' neighbors
    # 'indices[i]' are the face indices of those neighbors

    #--------------------------------------------------------------------------
    # 5) Build a weighted graph in NetworkX using centroid-distances as edges
    #--------------------------------------------------------------------------
    G = nx.Graph()
    # Add each face as a node in the graph
    G.add_nodes_from(range(num_faces))

    # For each face i, add edges (i -> j) for each neighbor j in the KNN
    for i in range(num_faces):
        for j, dist in zip(indices[i], distances[i]):
            if i == j:
                continue  # skip self-loop
            # Add an undirected edge with 'weight' = distance
            # NetworkX handles parallel edges gracefully via last add_edge,
            # but it typically overwrites the weight if (i, j) already exists.
            G.add_edge(i, j, weight=dist)

    #--------------------------------------------------------------------------
    # 6) Compute MST on that KNN graph
    #--------------------------------------------------------------------------
    mst = nx.minimum_spanning_tree(G, weight='weight')
    # Sort MST edges by ascending weight, so we add the shortest edges first
    mst_edges_sorted = sorted(
        mst.edges(data=True), key=lambda e: e[2]['weight']
    )
    print("mst edges sorted", len(mst_edges_sorted))
    #--------------------------------------------------------------------------
    # 7) Use a union-find structure to add MST edges only if they
    #    connect two currently disconnected components of the adjacency matrix
    #--------------------------------------------------------------------------

    # Convert face_adjacency to LIL format for efficient edge addition
    adjacency_lil = face_adjacency.tolil()

    # Now, step through MST edges in ascending order
    for (u, v, attr) in mst_edges_sorted:
        if uf.find(u) != uf.find(v):
            # These belong to different components, so unify them
            uf.union(u, v)
            # And add a "dummy" edge to our adjacency matrix
            adjacency_lil[u, v] = 1
            adjacency_lil[v, u] = 1

    # Convert back to CSR format and return
    face_adjacency = adjacency_lil.tocsr()

    if with_knn:
        print("Adding KNN edges.")
        ### Add KNN edges graph too
        dummy_row = []
        dummy_col = []
        for i in range(num_faces):
            for j in indices[i]:        
                dummy_row.extend([i, j])
                dummy_col.extend([j, i]) ### duplicates are handled by coo

        dummy_data = np.ones(len(dummy_row), dtype=np.int16)
        dummy_mat = coo_matrix(
            (dummy_data, (dummy_row, dummy_col)),
            shape=(num_faces, num_faces)
        ).tocsr()
        face_adjacency = face_adjacency + dummy_mat
        ###########################

    return face_adjacency

def construct_face_adjacency_matrix_naive(face_list):
    """
    Given a list of faces (each face is a 3-tuple of vertex indices),
    construct a face-based adjacency matrix of shape (num_faces, num_faces).
    Two faces are adjacent if they share an edge.

    If multiple connected components exist, dummy edges are added to 
    turn them into a single connected component. Edges are added naively by
    randomly selecting a face and connecting consecutive components -- (comp_i, comp_i+1) ...

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

    # Step 3: Ensure single connected component
    # Use connected_components to see how many components exist
    n_components, labels = connected_components(face_adjacency, directed=False)

    if n_components > 1:
        # We have multiple components; let's "connect" them via dummy edges
        # The simplest approach is to pick one face from each component
        # and connect them sequentially to enforce a single component.
        component_representatives = []

        for comp_id in range(n_components):
            # indices of faces in this component
            faces_in_comp = np.where(labels == comp_id)[0]
            if len(faces_in_comp) > 0:
                # take the first face in this component as a representative
                component_representatives.append(faces_in_comp[0])

        # Now, add edges between consecutive representatives
        dummy_row = []
        dummy_col = []
        for i in range(len(component_representatives) - 1):
            f_i = component_representatives[i]
            f_j = component_representatives[i + 1]
            dummy_row.extend([f_i, f_j])
            dummy_col.extend([f_j, f_i])

        if dummy_row:
            dummy_data = np.ones(len(dummy_row), dtype=np.int8)
            dummy_mat = coo_matrix(
                (dummy_data, (dummy_row, dummy_col)),
                shape=(num_faces, num_faces)
            ).tocsr()
            face_adjacency = face_adjacency + dummy_mat

    return face_adjacency

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

def solve_clustering(input_fname, uid, view_id, save_dir="test_results1", out_render_fol= "test_render_clustering", use_agglo=False, max_num_clusters=18, is_pc=False, option=1, with_knn=True, export_mesh=True):
    print(uid, view_id)
    
    if not is_pc:
        input_fname = f'{save_dir}/input_{uid}_{view_id}.ply'
        mesh = load_mesh_util(input_fname)

    else:
        pc = load_ply_to_numpy(input_fname)

    ### Load inferred PartField features
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

            
            pred_labels = np.zeros((len(labels), 1))
            for i, label in enumerate(np.unique(labels)):
                # print(i, label)
                pred_labels[labels == label] = i  # Assign RGB values to each label

            fname_clustering = os.path.join(out_render_fol, "cluster_out", str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2))
            np.save(fname_clustering, pred_labels)
            

            if not is_pc:
                V = mesh.vertices
                F = mesh.faces

                if export_mesh :
                    fname_mesh = os.path.join(out_render_fol, "ply", str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2) + ".ply")
                    export_colored_mesh_ply(V, F, pred_labels, filename=fname_mesh)

            
            else:
                if export_mesh:
                    fname_pc = os.path.join(out_render_fol, "ply", str(uid) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2) + ".ply")
                    export_pointcloud_with_labels_to_ply(pc, pred_labels, filename=fname_pc)
        
    else:
        if is_pc:
            print("Not implemented error. Agglomerative clustering only for mesh inputs.")
            exit()

        if option == 0:
            adj_matrix = construct_face_adjacency_matrix_naive(mesh.faces)
        elif option == 1:
            adj_matrix = construct_face_adjacency_matrix_facemst(mesh.faces, mesh.vertices, with_knn=with_knn)
        else:
            adj_matrix = construct_face_adjacency_matrix_ccmst(mesh.faces, mesh.vertices, with_knn=with_knn)

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
            relabel = np.zeros((len(FL), 1))
            for i, label in enumerate(unique_labels):
                relabel[FL == label] = i  # Assign RGB values to each label

            V = mesh.vertices
            F = mesh.faces

            if export_mesh :
                fname_mesh = os.path.join(out_render_fol, "ply", str(uid) + "_" + str(view_id) + "_" + str(max_num_clusters - n_cluster).zfill(2) + ".ply")
                export_colored_mesh_ply(V, F, FL, filename=fname_mesh)

            fname_clustering = os.path.join(out_render_fol, "cluster_out", str(uid) + "_" + str(view_id) + "_" + str(max_num_clusters - n_cluster).zfill(2))
            np.save(fname_clustering, FL)
        
        
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default= "", type=str)
    parser.add_argument('--root', default= "", type=str)
    parser.add_argument('--dump_dir', default= "", type=str)
    
    parser.add_argument('--max_num_clusters', default= 20, type=int)
    parser.add_argument('--use_agglo', default= False, type=bool)
    parser.add_argument('--is_pc', default= False, type=bool)
    parser.add_argument('--option', default= 1, type=int)
    parser.add_argument('--with_knn', default= False, type=bool)

    parser.add_argument('--export_mesh', default= True, type=bool)

    FLAGS = parser.parse_args()
    root = FLAGS.root
    OUTPUT_FOL = FLAGS.dump_dir
    SOURCE_DIR = FLAGS.source_dir

    MAX_NUM_CLUSTERS = FLAGS.max_num_clusters
    USE_AGGLO = FLAGS.use_agglo
    IS_PC = FLAGS.is_pc

    OPTION = FLAGS.option
    WITH_KNN = FLAGS.with_knn

    EXPORT_MESH = FLAGS.export_mesh

    models = os.listdir(root)
    os.makedirs(OUTPUT_FOL, exist_ok=True)

    cluster_fol = os.path.join(OUTPUT_FOL, "cluster_out")
    os.makedirs(cluster_fol, exist_ok=True) 

    if EXPORT_MESH:
        ply_fol = os.path.join(OUTPUT_FOL, "ply")
        os.makedirs(ply_fol, exist_ok=True)    

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
        if ".ply" in f and IS_PC and f.split(".")[0] not in existing_model_ids:
            selected.append(f)
        elif (".obj" in f or ".glb" in f) and not IS_PC and f.split(".")[0] not in existing_model_ids:
            selected.append(f)
    
    print("Number of models to process: " + str(len(selected)))
    
    for model in selected:
        fname = os.path.join(SOURCE_DIR, model)
        uid = model.split(".")[-2]
        view_id = 0

        solve_clustering(fname, uid, view_id, save_dir=root, out_render_fol= OUTPUT_FOL, use_agglo=USE_AGGLO, max_num_clusters=MAX_NUM_CLUSTERS, is_pc=IS_PC, option=OPTION, with_knn=WITH_KNN, export_mesh=EXPORT_MESH)
