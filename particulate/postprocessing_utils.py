import numpy as np
import trimesh


def refine_part_ids_strict(mesh, face_part_ids):
    """
    Refine face part IDs by treating each connected component (CC) in the mesh independently.
    For each CC, all faces are labeled with the part ID that has the largest surface area in that CC.
    
    Args:
        mesh: trimesh object
        face_part_ids: part ID for each face [num_faces]
    
    Returns:
        refined_face_part_ids: refined part ID for each face [num_faces]
    """
    face_part_ids = face_part_ids.copy()
    
    mesh_components = trimesh.graph.connected_components(
        edges=mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1
    )
    
    # For each connected component, find the part ID with the largest surface area
    for component in mesh_components:
        if len(component) == 0:
            continue
        
        part_id_areas = {}
        for face_idx in component:
            part_id = face_part_ids[face_idx]
            if part_id == -1:
                continue
            
            face_area = mesh.area_faces[face_idx]
            part_id_areas[part_id] = part_id_areas.get(part_id, 0.0) + face_area
        
        if len(part_id_areas) == 0:
            continue
        
        dominant_part_id = max(part_id_areas.keys(), key=lambda pid: part_id_areas[pid])
        
        for face_idx in component:
            face_part_ids[face_idx] = dominant_part_id
    
    return face_part_ids


def compute_part_components_for_mesh_cc(mesh, mesh_cc_faces, current_face_part_ids, face_adjacency_dict):
    """
    Compute part-specific connected components for faces in this mesh CC.
    
    Two faces are in the same component if:
    - They have the same part ID
    - They are connected through faces of the same part ID
    
    Args:
        mesh: trimesh object
        mesh_cc_faces: array of face indices belonging to this mesh connected component
        current_face_part_ids: part ID for each face [num_faces]
        face_adjacency_dict: dictionary mapping face index to set of adjacent face indices
    
    Returns:
        components: list of dicts, each with keys 'faces' (array of face indices), 
                   'part_id' (int), and 'area' (float)
    """
    components = []
    
    # Get unique part IDs in this mesh CC
    unique_part_ids = np.unique(current_face_part_ids[mesh_cc_faces])
    
    for part_id in unique_part_ids:
        if part_id == -1:
            continue
        
        # Get faces in this mesh CC with this part ID
        mask = current_face_part_ids[mesh_cc_faces] == part_id
        faces_with_part = mesh_cc_faces[mask]
        
        if len(faces_with_part) == 0:
            continue
        
        # Convert to set for faster lookup
        faces_with_part_set = set(faces_with_part)
        
        # Build edges between these faces (both must have same part ID and be adjacent)
        edges_for_part = []
        for face_i in faces_with_part:
            for face_j in face_adjacency_dict[face_i]:
                if face_j in faces_with_part_set:
                    edges_for_part.append([face_i, face_j])
        
        if len(edges_for_part) == 0:
            # Each face is its own component
            for face_i in faces_with_part:
                components.append({
                    'faces': np.array([face_i]),
                    'part_id': part_id,
                    'area': mesh.area_faces[face_i]
                })
        else:
            # Find connected components
            edges_for_part = np.array(edges_for_part)
            comps = trimesh.graph.connected_components(
                edges=edges_for_part,
                nodes=faces_with_part,
                min_len=1
            )
            
            for comp in comps:
                comp_faces = np.array(list(comp))
                components.append({
                    'faces': comp_faces,
                    'part_id': part_id,
                    'area': np.sum(mesh.area_faces[comp_faces])
                })
    
    return components


def refine_part_ids_nonstrict(mesh, face_part_ids):
    """
    Refine face part IDs to ensure each part ID forms a single connected component.
    
    For each part ID, if there are multiple disconnected components, the smaller
    components (by surface area) are reassigned based on adjacent faces' part IDs.
    This is done iteratively until convergence.
    
    Args:
        mesh: trimesh object
        face_part_ids: initial part ID for each face [num_faces]
    
    Returns:
        face_part_ids_final: refined part IDs using iterative refinement 
              (ensures each part ID forms a single connected component) [num_faces]
    """
    face_part_ids_final = face_part_ids.copy()
    
    # Step 1: Find connected components of the original mesh (immutable structure)
    mesh_components = trimesh.graph.connected_components(
        edges=mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1
    )
    mesh_components = [np.array(list(comp)) for comp in mesh_components]
    
    # Step 2: Build face adjacency dict (immutable structure)
    face_adjacency_dict = {i: set() for i in range(len(mesh.faces))}
    for face_i, face_j in mesh.face_adjacency:
        face_adjacency_dict[face_i].add(face_j)
        face_adjacency_dict[face_j].add(face_i)
    
    # Step 3: Process each mesh CC independently
    for mesh_cc_faces in mesh_components:
        done = False
        while not done:
            comps = compute_part_components_for_mesh_cc(mesh, mesh_cc_faces, face_part_ids_final, face_adjacency_dict)
            comps.sort(key=lambda c: c['area'])

            part_id_areas = {}
            for comp in comps:
                pid = comp['part_id']
                if pid not in part_id_areas:
                    part_id_areas[pid] = 0.0
                part_id_areas[pid] += comp['area']

            done = True
            for comp_idx in range(len(comps)):
                current_part_id = comps[comp_idx]['part_id']
                if len([c for c in comps if c['part_id'] == current_part_id]) > 1:
                    done = False
                    # Find adjacent components
                    adjacent_part_ids = set()
                    current_faces_set = set(comps[comp_idx]['faces'])

                    for face_i in current_faces_set:
                        for face_j in face_adjacency_dict[face_i]:
                            if face_j in current_faces_set:
                                continue
                            adjacent_part_ids.add(face_part_ids_final[face_j])
                    
                    chosen_part_id = max(adjacent_part_ids, key=lambda x: part_id_areas[x])
                    comps[comp_idx]['part_id'] = chosen_part_id
                    face_part_ids_final[comps[comp_idx]['faces']] = chosen_part_id
                    break
    
    return face_part_ids_final


def find_part_ids_for_faces(mesh, part_ids, face_indices, strict=False):
    """
    Assign part IDs to each face in the mesh based on sampled points.
    
    For each face, uses majority vote from all points that lie on that face.
    Then applies refinement to ensure consistent part labeling.
    
    Args:
        mesh: trimesh object
        part_ids: part IDs for each sampled point [num_points]
        face_indices: which face each point lies on (-1 means on edge) [num_points]
        strict: whether to use strict refinement (each mesh CC gets the dominant part ID)
    
    Returns:
        tuple: (face_part_ids, face_part_ids_refined_strict, face_part_ids_refined)
            - face_part_ids: initial part IDs assigned via majority vote [num_faces]
            - face_part_ids_refined_strict: refined part IDs using strict refinement 
              (each mesh CC gets the dominant part ID) [num_faces]
            - face_part_ids_refined: refined part IDs using iterative refinement 
              (ensures each part ID forms a single connected component) [num_faces]
    """
    num_faces = len(mesh.faces)
    face_part_ids = np.full(num_faces, -1, dtype=np.int32)
    
    # For each face, collect all points that lie on it and use majority vote
    face_to_points = {}
    for point_idx, face_idx in enumerate(face_indices):
        if face_idx == -1:  # Point is on an edge, ignore it
            continue
        if face_idx not in face_to_points:
            face_to_points[face_idx] = []
        face_to_points[face_idx].append(part_ids[point_idx])
    
    # Assign part IDs based on majority vote from points
    for face_idx, point_part_ids in face_to_points.items():
        # Use bincount to find the majority part ID
        counts = np.bincount(point_part_ids)
        majority_part_id = np.argmax(counts)
        face_part_ids[face_idx] = majority_part_id
    
    if strict:
        return refine_part_ids_strict(mesh, face_part_ids)
    else:
        return refine_part_ids_nonstrict(mesh, face_part_ids)
