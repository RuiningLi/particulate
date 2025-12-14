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

    """System Options"""
    device: str = "cuda"  #  Device
    debug: bool = False  #  enable debug checks
    extras: bool = False # include extra output for viz/debugging

    """ State """
    mode: str = 'pca'
    m: dict = None          # mesh

    # pca mode

    # feature explore mode
    i_feature: int = 0

    i_cluster: int = 1 

    i_eps: int = 0.6 

    ### For mixing in clustering
    weight_dist = 1.0
    weight_feat = 1.0
    
    ### For clustering visualization
    feature_range: float = 0.1
    continuous_explore: bool = False

    viz_mode: str = "faces"

    output_fol: str = "results_single"

    ### For adj_matrix
    adj_mode: str = "Vanilla"
    add_knn_edges: bool = False

    ### counter for screenshot
    counter: int = 0

modes_list = ['pca', 'feature_viz', 'cluster_agglo', 'cluster_kmeans']
adj_mode_list = ["Vanilla", "Face_MST", "CC_MST"]

#### For clustering
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

#####################################
## Face adjacency computation options
#####################################
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
#####################################

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

    print(F)
    print(V[F[1][0]])
    print(V[F[1][1]])
    print(V[F[1][2]])

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

    if opts.mode == 'pca':
        psim.TextUnformatted("""3-dim PCA embeddeding of features is shown as rgb color""")
        viz_pca_colors(m)

    elif opts.mode == 'feature_viz':
        psim.TextUnformatted("""Use the slider to scrub through all features.\nCtrl-click to type a particular index.""")

        this_changed, opts.i_feature = psim.SliderInt("feature index", opts.i_feature, v_min=0, v_max=(m['feat_np'].shape[-1]-1))
        this_changed = this_changed or changed

        if this_changed:
            viz_feature(m, opts.i_feature)
    
    elif opts.mode == "cluster_agglo":
        psim.TextUnformatted("""Use the slider to toggle the number of desired clusters.""")
        cluster_changed, opts.i_cluster = psim.SliderInt("number of clusters", opts.i_cluster, v_min=1, v_max=30)

        ### To handle different face adjacency options
        mode_changed, ind = psim.Combo("Adj Matrix Def", adj_mode_list.index(opts.adj_mode), adj_mode_list)
        knn_changed, opts.add_knn_edges = psim.Checkbox("Add KNN edges", opts.add_knn_edges)
        
        if mode_changed:
            opts.adj_mode = adj_mode_list[ind]

        if psim.Button("Recompute"):

            ### Run clustering algorithm
            num_clusters = opts.i_cluster

            ### Mesh 1
            point_feat = m['feat_np']
            point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
            
            ### Compute adjacency matrix ###
            if opts.adj_mode == "Vanilla":
                adj_matrix = construct_face_adjacency_matrix_naive(opts.m["F"])
            elif opts.adj_mode == "Face_MST":
                adj_matrix = construct_face_adjacency_matrix_facemst(opts.m["F"], opts.m["V"], with_knn=opts.add_knn_edges)
            elif opts.adj_mode == "CC_MST":
                adj_matrix = construct_face_adjacency_matrix_ccmst(opts.m["F"], opts.m["V"], with_knn=opts.add_knn_edges)            
            ################################

            ## Agglomerative clustering
            clustering = AgglomerativeClustering(connectivity= adj_matrix,
                                        n_clusters=num_clusters,
                                        ).fit(point_feat)

            m['ps_mesh'].add_scalar_quantity("cluster", clustering.labels_, cmap='turbo', vminmax=(0, num_clusters-1), enabled=True, defined_on=m["viz_mode"])
            print("Recomputed.")      


    elif opts.mode == "cluster_kmeans":
        psim.TextUnformatted("""Use the slider to toggle the number of desired clusters.""")

        cluster_changed, opts.i_cluster = psim.SliderInt("number of clusters", opts.i_cluster, v_min=1, v_max=30)

        if psim.Button("Recompute"):

            ### Run clustering algorithm
            num_clusters = opts.i_cluster

            ### Mesh 1
            point_feat = m['feat_np']
            point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
            clustering = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(point_feat)

            m['ps_mesh'].add_scalar_quantity("cluster", clustering.labels_, cmap='turbo', vminmax=(0, num_clusters-1), enabled=True, defined_on=m["viz_mode"])

def main():
    ## Parse args
    # Uses simple_parsing library to automatically construct parser from the dataclass Options
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    parser.add_argument('--data_root', default="../exp_results/partfield_features/trellis/", help='Path the model features are stored.')
    args = parser.parse_args()
    opts: Options = args.options

    DATA_ROOT = args.data_root

    shape_1 = opts.filename

    if os.path.exists(os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0.npy")):
        feature_fname1 = os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0.npy")
        mesh_fname1 = os.path.join(DATA_ROOT, "feat_pca_"+ shape_1 + "_0.ply")
    else:
        feature_fname1 = os.path.join(DATA_ROOT, "part_feat_"+ shape_1 + "_0_batch.npy")
        mesh_fname1 = os.path.join(DATA_ROOT, "feat_pca_"+ shape_1 + "_0.ply")

    #### To save output ####
    os.makedirs(opts.output_fol, exist_ok=True)      
    ########################

    # Initialize
    ps.init()

    mesh_dict = load_features(feature_fname1, mesh_fname1, opts.viz_mode)
    prep_feature_mesh(mesh_dict)
    mesh_dict["viz_mode"] = opts.viz_mode
    opts.m = mesh_dict

    # Start the interactive UI
    ps.set_user_callback(lambda : ps_callback(opts))
    ps.show()


if __name__ == "__main__":
    main()

