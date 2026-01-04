import numpy as np
from typing import List, Tuple, Optional, Union


def axis_point_to_plucker(axis: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Convert axis-point coordinates to plucker coordinates.
    """
    assert axis.shape[-1] == 3
    assert point.shape[-1] == 3
    l = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8)
    m = np.cross(l, point, axis=-1)
    return np.concatenate([l, m], axis=-1)


def plucker_to_axis_point(plucker: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert plucker coordinates to axis-point coordinates.
    """
    assert plucker.shape[-1] == 6
    l, m = plucker[..., :3], plucker[..., 3:]
    axis = l / (np.linalg.norm(l, axis=-1, keepdims=True) + 1e-8)
    point = np.cross(m, axis, axis=-1)
    return axis, point


def plucker_to_4x4_transform_matrix(plucker: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert plucker coordinates to a 4x4 transformation matrix.
    """
    assert plucker.shape == (6,)
    axis, point = plucker_to_axis_point(plucker)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = point - R @ point
    
    return T


def shift_axes_plucker(
    axes_plucker: np.ndarray,
    x_center: float, y_center: float, z_center: float, scale: float
) -> np.ndarray:
    """Shift the axes plucker coordinates."""
    axis, point = plucker_to_axis_point(axes_plucker)
    point_new = point - np.array([x_center, y_center, z_center])
    point_new *= scale
    return axis_point_to_plucker(axis, point_new)


def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points by a 4x4 transformation matrix.

    points: (..., 3)
    transform_matrix: (4, 4)
    """
    return points @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]


def transform_direction(direction: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a direction vector by a 4x4 transformation matrix.

    direction: (..., 3)
    transform_matrix: (4, 4)
    """
    return direction @ transform_matrix[:3, :3].T


def get_subtree_part_ids(motion_hierarchy: List[Tuple[int, int]], part_id: int) -> List[int]:
    """
    Get the subtree part ids for a given part id.
    """
    subtree_part_ids = [part_id]
    for parent_id, child_id in motion_hierarchy:
        if parent_id == part_id:
            subtree_part_ids.extend(get_subtree_part_ids(motion_hierarchy, child_id))
    return subtree_part_ids


def get_part_order_from_root(motion_hierarchy: List[Tuple[int, int]]) -> List[int]:
    """
    Depth-first search to get the part order from the root.
    """
    part_order = []
    visited = set()
    def dfs(part_id):
        if part_id in visited:
            return
        part_order.append(part_id)
        visited.add(part_id)
        for parent_id, child_id in motion_hierarchy:
            if parent_id == part_id:
                dfs(child_id)

    # Find the base/root part id
    all_part_ids = set([parent_id for parent_id, _ in motion_hierarchy])
    all_part_ids.update([child_id for _, child_id in motion_hierarchy])

    # Find the root part id
    for _, child_id in motion_hierarchy:
        all_part_ids.remove(child_id)

    # assert len(all_part_ids) == 1
    root_part_id = all_part_ids.pop()

    dfs(root_part_id) # Populate part_order
    return part_order


def articulate_points(
    xyz: np.ndarray,
    part_ids: np.ndarray,
    motion_hierarchy: List[Tuple[int, int]],
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    articulation_state: Union[float, np.ndarray],  # Value between 0 and 1
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Articulate points based on given articulation state.
    
    Args:
        articulation_state: Value between 0 (low limit) and 1 (high limit)
    
    Returns:
        Tuple of (articulated_xyz, articulated_revolute_plucker, articulated_prismatic_axis)
    """
    articulated_xyz = xyz.copy()
    articulated_revolute_plucker = revolute_plucker.copy()
    articulated_prismatic_axis = prismatic_axis.copy()
    articulated_revolute_range = revolute_range.copy()
    articulated_prismatic_range = prismatic_range.copy()
    if normals is not None:
        articulated_normals = normals.copy()

    if len(motion_hierarchy) == 0:
        return articulated_xyz, articulated_revolute_plucker, articulated_prismatic_axis
    
    part_order = get_part_order_from_root(motion_hierarchy)
    
    for pid in part_order:
        affected_part_ids = get_subtree_part_ids(motion_hierarchy, pid)
        
        if is_part_revolute[pid]:
            low_limit, high_limit = revolute_range[pid]
            part_articulation_state = articulation_state if isinstance(articulation_state, float) else articulation_state[pid]
            # Interpolate between low and high limits
            angle = low_limit + part_articulation_state * (high_limit - low_limit)
            articulated_revolute_range[pid] = np.array([low_limit - angle, high_limit - angle])
            transform_matrix = plucker_to_4x4_transform_matrix(articulated_revolute_plucker[pid], angle)
            
            for affected_pid in affected_part_ids:
                # Transform points
                articulated_xyz[part_ids == affected_pid] = transform_points(
                    articulated_xyz[part_ids == affected_pid], transform_matrix
                )
                # Transform normals
                if normals is not None:
                    articulated_normals[part_ids == affected_pid] = transform_direction(
                        articulated_normals[part_ids == affected_pid], transform_matrix
                    )
                
                # Transform revolute axes for affected parts
                if is_part_revolute[affected_pid]:
                    current_axis, current_point = plucker_to_axis_point(articulated_revolute_plucker[affected_pid])
                    new_axis = transform_direction(current_axis, transform_matrix)
                    new_point = transform_points(current_point, transform_matrix)
                    articulated_revolute_plucker[affected_pid] = axis_point_to_plucker(new_axis, new_point)
                
                # Transform prismatic axes for affected parts
                if is_part_prismatic[affected_pid]:
                    articulated_prismatic_axis[affected_pid] = transform_direction(
                        articulated_prismatic_axis[affected_pid], transform_matrix
                    )
        
        if is_part_prismatic[pid]:
            low_limit, high_limit = prismatic_range[pid]
            part_articulation_state = articulation_state if isinstance(articulation_state, float) else articulation_state[pid]
            # Interpolate between low and high limits
            displacement = low_limit + part_articulation_state * (high_limit - low_limit)
            articulated_prismatic_range[pid] = np.array([low_limit - displacement, high_limit - displacement])
            paxis = articulated_prismatic_axis[pid]
            
            for affected_pid in affected_part_ids:
                # Translate points
                articulated_xyz[part_ids == affected_pid] = (
                    articulated_xyz[part_ids == affected_pid] + displacement * paxis
                )
                # Translation does not affect normals
                
                # Translate revolute axes for affected parts (only the point, not the direction)
                if is_part_revolute[affected_pid]:
                    current_axis, current_point = plucker_to_axis_point(articulated_revolute_plucker[affected_pid])
                    new_point = current_point + displacement * paxis
                    articulated_revolute_plucker[affected_pid] = axis_point_to_plucker(current_axis, new_point)
    
    output = (articulated_xyz, articulated_revolute_plucker, articulated_prismatic_axis, articulated_revolute_range, articulated_prismatic_range)
    if normals is not None:
        output = output + (articulated_normals,)
    return output


def compute_part_transforms(
    unique_part_ids,
    motion_hierarchy,
    is_part_revolute,
    is_part_prismatic,
    revolute_plucker,
    revolute_range,
    prismatic_axis,
    prismatic_range,
    articulation_state
):
    """
    Compute the 4x4 transformation matrix for each part at a given articulation state.
    Returns a dictionary mapping part_id to its cumulative transformation matrix.
    
    The transformation represents how to transform each part from its rest pose to the articulated pose.
    """
    if len(motion_hierarchy) == 0:
        return {pid: np.eye(4) for pid in unique_part_ids}
    
    # Collect all relevant part IDs from motion hierarchy and unique_part_ids
    all_part_ids = set(unique_part_ids)
    for parent, child in motion_hierarchy:
        all_part_ids.add(parent)
        all_part_ids.add(child)
        
    transforms = {pid: np.eye(4) for pid in all_part_ids}
    
    # Process parts in hierarchical order (BFS/DFS from root)
    part_order = get_part_order_from_root(motion_hierarchy)
    
    for pid in part_order:
        affected_part_ids = get_subtree_part_ids(motion_hierarchy, pid)
        
        # Compute transformation for this part's joint
        joint_transform = np.eye(4)
        
        if is_part_revolute[pid]:
            low_limit, high_limit = revolute_range[pid]
            angle = low_limit + articulation_state * (high_limit - low_limit)
            joint_transform = plucker_to_4x4_transform_matrix(revolute_plucker[pid], angle)
        
        elif is_part_prismatic[pid]:
            low_limit, high_limit = prismatic_range[pid]
            displacement = low_limit + articulation_state * (high_limit - low_limit)
            paxis = prismatic_axis[pid]
            joint_transform[:3, 3] = displacement * paxis
        
        # Apply joint transformation to all affected (descendant) parts
        for affected_pid in affected_part_ids:
            if affected_pid in transforms:
                transforms[affected_pid] = joint_transform @ transforms[affected_pid]
    
    return transforms


def closest_point_on_axis_to_revolute_plucker(
    closest_point_on_axis: np.ndarray,
    part_ids: np.ndarray,
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_axis: np.ndarray,
) -> np.ndarray:
    """
    Convert closest point on axis to motion parameters.
    """
    num_parts = revolute_axis.shape[0]
    revolute_plucker = np.zeros((num_parts, 6))
    for pid in np.unique(part_ids):
        if is_part_revolute[pid]:
            closest_points_on_axis = closest_point_on_axis[part_ids == pid]
            current_revolute_axis = revolute_axis[pid]
            selected_point = np.median(closest_points_on_axis, axis=0)
            plucker = axis_point_to_plucker(current_revolute_axis, selected_point)
            revolute_plucker[pid, :] = plucker
    return revolute_plucker


def articulate_mesh_parts(
    mesh_parts,
    unique_part_ids,
    motion_hierarchy,
    is_part_revolute, is_part_prismatic,
    revolute_plucker, revolute_range,
    prismatic_axis, prismatic_range,
    articulation_state
):
    """
    Articulate mesh parts based on given articulation state.
    """
    all_verts = [mesh_parts[i].vertices for i in range(len(mesh_parts))]
    all_part_ids = [np.full(len(mesh_parts[i].vertices), unique_part_ids[i], dtype=np.int32) for i in range(len(mesh_parts))]

    verts_transformed = articulate_points(
        xyz=np.concatenate(all_verts, axis=0),
        part_ids=np.concatenate(all_part_ids, axis=0),
        motion_hierarchy=motion_hierarchy,
        is_part_revolute=is_part_revolute,
        is_part_prismatic=is_part_prismatic,
        revolute_plucker=revolute_plucker,
        revolute_range=revolute_range,
        prismatic_axis=prismatic_axis,
        prismatic_range=prismatic_range,
        articulation_state=articulation_state
    )[0]

    mesh_parts_articulated = [mesh_parts[i].copy() for i in range(len(mesh_parts))]
    vert_offset = 0
    for i in range(len(mesh_parts)):
        mesh_parts_articulated[i].vertices = verts_transformed[vert_offset:vert_offset + len(mesh_parts[i].vertices)]
        vert_offset += len(mesh_parts[i].vertices)
    return mesh_parts_articulated

def articulate_bbox(
    bbox_vertices: np.ndarray,
    part_ids: np.ndarray,
    motion_hierarchy: List[Tuple[int, int]],
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    articulation_state: Union[float, np.ndarray],  # Value between 0 and 1
    rotation_origin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Articulate bbox vertices based on given articulation state.
    
    Args:
        articulation_state: Value between 0 (low limit) and 1 (high limit)
    
    Returns:
        Tuple of (articulated_xyz, articulated_revolute_plucker, articulated_prismatic_axis)
    """
    articulated_bbox_vertices = bbox_vertices.copy()
    articulated_revolute_plucker = revolute_plucker.copy()
    articulated_prismatic_axis = prismatic_axis.copy()
    articulated_revolute_range = revolute_range.copy()
    articulated_prismatic_range = prismatic_range.copy()

    part_transformations = [[] for _ in range(len(np.unique(part_ids)))]
    if len(motion_hierarchy) == 0:
        return part_transformations
    
    part_order = get_part_order_from_root(motion_hierarchy)
    # reverse part order
    part_order = part_order[::-1]

    for pid in part_order:
        if pid not in part_ids:
            continue
        affected_part_ids = get_subtree_part_ids(motion_hierarchy, pid)
        applied_tramsformation_matrix = np.eye(4, dtype=np.float32)
        applied_rotation_axis_origin = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        applied_transformation_type = "none"
        children_idxs = affected_part_ids

        if is_part_revolute[pid]:
            low_limit, high_limit = articulated_revolute_range[pid]
            part_articulation_state = articulation_state if isinstance(articulation_state, float) else articulation_state[pid]
            # Interpolate between low and high limits
            angle = part_articulation_state * (high_limit - low_limit) # check if this is radian 
            axis, origin = plucker_to_axis_point(articulated_revolute_plucker[pid])
            transform_matrix = plucker_to_4x4_transform_matrix(articulated_revolute_plucker[pid], angle)
            rotation_axis_origin = origin if rotation_origin is None else rotation_origin[pid]
            applied_tramsformation_matrix[:3, :3] = transform_matrix[:3, :3]
            applied_rotation_axis_origin = rotation_axis_origin
            applied_transformation_type = "rotation"
            
        
        elif is_part_prismatic[pid]:
            low_limit, high_limit = prismatic_range[pid]
            part_articulation_state = articulation_state if isinstance(articulation_state, float) else articulation_state[pid]
            # Interpolate between low and high limits
            displacement = part_articulation_state * (high_limit - low_limit)
            articulated_prismatic_range[pid] = np.array([low_limit - displacement, high_limit - displacement])
            paxis = articulated_prismatic_axis[pid]

            translation = paxis * displacement
            applied_tramsformation_matrix[:3, 3] = translation
            applied_transformation_type = "translation"
        
        if not applied_transformation_type == "none":
            record = {
                "type": applied_transformation_type,
                "matrix": applied_tramsformation_matrix,
                "rotation_axis_origin": applied_rotation_axis_origin
            }
            for child_idx in list(set([pid] + children_idxs)):
                # part_transformations[child_idx].append(record)
                # turn part_id to index
                if child_idx not in part_ids:
                    continue
                try:
                    child_idx_index = np.where(np.unique(part_ids) == child_idx)[0][0]
                    part_transformations[child_idx_index].append(record)
                except:
                    breakpoint()


    return part_transformations
