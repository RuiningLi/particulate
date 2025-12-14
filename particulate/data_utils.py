from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_obj_raw_preserve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from an OBJ file while preserving vertex order.

    Args:
        path (Path): Path to the OBJ file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - vertices: Nx3 array of vertex positions
            - faces: Mx3 array of face indices (0-based)
    """
    verts, faces = [], []
    with path.open() as fh:
        for ln in fh:
            if ln.startswith('v '):   # keep order *exactly* as file
                _, x, y, z = ln.split()[:4]
                verts.append([float(x), float(y), float(z)])
            elif ln.startswith('f '):
                toks = ln[2:].strip().split()
                if len(toks) == 3:
                    faces.append([int(t.split('/')[0]) - 1 for t in toks])
                else:
                    faces.append([int(t.split('/')[0]) - 1 for t in toks[:3]])
                    for i in range(2, len(toks) - 1):
                        faces.append([int(toks[0].split('/')[0]) - 1,
                                    int(toks[i].split('/')[0]) - 1,
                                    int(toks[i + 1].split('/')[0]) - 1])
    return np.asarray(verts, float), np.asarray(faces, int)


def sharp_sample_pointcloud(mesh, num_points: int = 8192):
    V = mesh.vertices
    N = mesh.face_normals
    F = mesh.faces
    
    edge_to_faces = {}
    
    for face_idx in range(len(F)):
        face = F[face_idx]
        edges = [
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0])
        ]
        
        for edge in edges:
            edge_key = tuple(sorted(edge))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append(face_idx)
    
    sharp_edges = []
    sharp_edge_normals = []
    sharp_edge_faces = []
    cos_30 = np.cos(np.radians(30))  # ≈ 0.866
    cos_150 = np.cos(np.radians(150))  # ≈ -0.866
    
    for edge_key, face_indices in edge_to_faces.items():
        if len(face_indices) < 2:
            continue
        
        is_sharp = False
        for i in range(len(face_indices)):
            for j in range(i + 1, len(face_indices)):
                n1 = N[face_indices[i]]
                n2 = N[face_indices[j]]
                dot_product = np.dot(n1, n2)
                
                if cos_150 < dot_product < cos_30 and np.linalg.norm(n1) > 1e-8 and np.linalg.norm(n2) > 1e-8:
                    is_sharp = True
                    sharp_edges.append(edge_key)
                    averaged_normal = (n1 + n2) / 2
                    sharp_edge_normals.append(averaged_normal)
                    sharp_edge_faces.append(face_indices)  # Store all adjacent faces
                    break
            if is_sharp:
                break
    
    edge_a = np.array([edge[0] for edge in sharp_edges], dtype=np.int32)
    edge_b = np.array([edge[1] for edge in sharp_edges], dtype=np.int32)
    sharp_edge_normals = np.array(sharp_edge_normals, dtype=np.float64)

    if len(sharp_edges) == 0:
        samples = np.zeros((0, 3), dtype=np.float64)
        normals = np.zeros((0, 3), dtype=np.float64)
        edge_indices = np.zeros((0,), dtype=np.int32)
        return samples, normals, edge_indices, sharp_edge_faces

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num_points)
    w = np.random.rand(num_points, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = sharp_edge_normals[index]
    return samples, normals, index, sharp_edge_faces


def sample_points(mesh, num_points, sharp_point_ratio, at_least_one_point_per_face=False):
    """Sample points from mesh using sharp edge and uniform sampling."""
    num_points_sharp_edges = int(num_points * sharp_point_ratio)
    num_points_uniform = num_points - num_points_sharp_edges
    points_sharp, normals_sharp, edge_indices, sharp_edge_faces = sharp_sample_pointcloud(mesh, num_points_sharp_edges)

    # If no sharp edges were found, sample all points uniformly
    if len(points_sharp) == 0 and sharp_point_ratio > 0:
        print(f"Warning: No sharp edges found, sampling all points uniformly")
        num_points_uniform = num_points

    if at_least_one_point_per_face:
        num_faces = len(mesh.faces)
        if num_points_uniform < num_faces:
            raise ValueError(
                "Unable to sample at least one point per face: "
                f"{num_faces} faces > {num_points_uniform}"
            )
        
        face_perm = np.random.permutation(num_faces)
        points_per_face = []
        for face_idx in face_perm:
            r1, r2 = np.random.random(), np.random.random()
            sqrt_r1 = np.sqrt(r1)
            u = 1 - sqrt_r1
            v = sqrt_r1 * (1 - r2)
            w = sqrt_r1 * r2
            
            face = mesh.faces[face_idx]
            vertices = mesh.vertices[face]
            
            point = u * vertices[0] + v * vertices[1] + w * vertices[2]
            points_per_face.append(point)
        
        points_per_face = np.array(points_per_face)
        normals_per_face = mesh.face_normals[face_perm]
        
        num_remaining_points = num_points_uniform - num_faces
        if num_remaining_points > 0:
            points_remaining, face_indices_remaining = mesh.sample(num_remaining_points, return_index=True)
            normals_remaining = mesh.face_normals[face_indices_remaining]
            
            points_uniform = np.concatenate([points_per_face, points_remaining], axis=0)
            normals_uniform = np.concatenate([normals_per_face, normals_remaining], axis=0)
            face_indices = np.concatenate([face_perm, face_indices_remaining], axis=0)
        else:
            points_uniform = points_per_face
            normals_uniform = normals_per_face
            face_indices = face_perm
    else:
        points_uniform, face_indices = mesh.sample(num_points_uniform, return_index=True)
        normals_uniform = mesh.face_normals[face_indices]

    points = np.concatenate([points_sharp, points_uniform], axis=0)
    normals = np.concatenate([normals_sharp, normals_uniform], axis=0)
    sharp_flag = np.concatenate([
        np.ones(len(points_sharp), dtype=np.bool_),
        np.zeros(len(points_uniform), dtype=np.bool_)
    ], axis=0)
    
    # For each sharp point, randomly select one of the adjacent faces from the edge
    sharp_face_indices = np.zeros(len(points_sharp), dtype=np.int32)
    for i, edge_idx in enumerate(edge_indices):
        adjacent_faces = sharp_edge_faces[edge_idx]
        # Randomly select one of the adjacent faces
        sharp_face_indices[i] = np.random.choice(adjacent_faces)
    
    face_indices = np.concatenate([
        sharp_face_indices,
        face_indices
    ], axis=0)
    
    return points, normals, sharp_flag, face_indices
