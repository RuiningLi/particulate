import numpy as np
from typing import List, Mapping, Tuple, Dict
from scipy.optimize import linear_sum_assignment
import torch

from particulate.articulation_utils import articulate_points, articulate_bbox
from pytorch3d.loss import chamfer_distance


def hungarian_matching_cdist(points1, part_ids1, points2, part_ids2, cost_type="cdist"):
    """
    Find the correspondences from the first object to the second object based on closest bbox centers using Hungarian algorithm\n

    - points1: point cloud for first object
    - part_ids1: part IDs for first object
    - points2: point cloud for second object  
    - part_ids2: part IDs for second object

    Return:\n
    - mapping: the mapping from the first object to the second object in the form: [[obj_part_idx], ...]
    """
    INF = 9999999

    unique_parts1 = np.unique(part_ids1)
    unique_parts2 = np.unique(part_ids2)
    n_parts1 = len(unique_parts1)
    n_parts2 = len(unique_parts2)

    # Compute centers for each part
    centers_1 = {}
    centers_2 = {}
    
    for part_id in unique_parts1:
        part_points = points1[part_ids1 == part_id]
        centers_1[part_id] = np.mean(part_points, axis=0)
    
    for part_id in unique_parts2:
        part_points = points2[part_ids2 == part_id]
        centers_2[part_id] = np.mean(part_points, axis=0)
    # Initialize the cost matrix
    cost_matrix = np.ones((n_parts1, n_parts2), dtype=np.float32) * INF
    for i in range(n_parts1):
        for j in range(n_parts2):
            if cost_type == "cdist":
                cost_matrix[i, j] = np.linalg.norm(
                    centers_1[unique_parts1[i]] - centers_2[unique_parts2[j]]
                )
            elif cost_type == "chamfer":
                pred_points = points1[part_ids1 == unique_parts1[i]]
                gt_points = points2[part_ids2 == unique_parts2[j]]
                random_int_pred = np.random.randint(0, len(pred_points), min(len(pred_points), 1000))
                random_int_gt = np.random.randint(0, len(gt_points), min(len(gt_points), 1000))
                sampled_pred_points = pred_points[random_int_pred]
                sampled_gt_points = gt_points[random_int_gt]
                cost_matrix[i, j] = chamfer_distance(torch.from_numpy(sampled_pred_points)[None, :, :].float(), torch.from_numpy(sampled_gt_points)[None, :, :].float())[0].item()
            else:
                raise ValueError(f"Unknown cost type: {cost_type}")

    # Find the correspondences using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Construct the mapping - initialize all prediction parts as unmatched
    mapping = np.full((n_parts1, 2), -1, dtype=np.float32)
    
    # Assign matched pairs (one-to-one matching)
    for i in range(len(row_ind)):
        row_idx = row_ind[i]  # Index in unique_parts1
        col_idx = col_ind[i]  # Index in unique_parts2
        cost = cost_matrix[row_idx, col_idx]
        
        # Only assign if cost is reasonable (not INF)
        if cost < INF:
            # row_idx is already the index in unique_parts1
            mapping[row_idx, 0] = col_idx
            mapping[row_idx, 1] = cost                    # Distance cost

    return mapping

def _sample_points_in_box3d(bbox_vertices, num_samples):
    """
    Sample points in a axis-aligned 3D bounding box\n
    - bbox_vertices: the vertices of the bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - num_samples: the number of samples to use\n

    Return:\n
    - points: the sampled points in the form: [[x0, y0, z0], [x1, y1, z1], ...]
    """

    # Compute the bounding box size
    bbox_size = np.max(bbox_vertices, axis=0) - np.min(bbox_vertices, axis=0)

    # Sample points in the bounding box
    points = np.random.rand(num_samples, 3) * bbox_size + np.min(bbox_vertices, axis=0)

    return points

def _apply_forward_transformations(points, transformations):
    """
    Reference: https://github.com/3dlg-hcvc/singapo/blob/main/metrics/giou.py
    Apply forward transformations to the points\n
    - points: the points in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - transformations: list of transformations to apply\n

    Return:\n
    - points_transformed: the transformed points in the form: [[x0, y0, z0], [x1, y1, z1], ...]
    """
    if len(transformations) == 0:
        return points

    # To homogeneous coordinates
    points_transformed = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # Apply the transformations one by one in order
    for transformation in transformations:
        if transformation["type"] == "translation":
            points_transformed = np.matmul(
                transformation["matrix"], points_transformed.T
            ).T

        elif transformation["type"] == "rotation":
            axis_origin = np.append(transformation["rotation_axis_origin"], 0)
            points_recentered = points_transformed - axis_origin

            points_rotated = np.matmul(transformation["matrix"], points_recentered.T).T
            points_transformed = points_rotated + axis_origin

        elif transformation["type"] == "plucker":
            points_transformed = np.matmul(
                transformation["matrix"], points_transformed.T
            ).T

        else:
            raise ValueError(f"Unknown transformation type: {transformation['type']}")

    return points_transformed[..., :3]


def _apply_backward_transformations(points, transformations):
    """
    Reference: https://github.com/3dlg-hcvc/singapo/blob/main/metrics/iou_cdist.py

    Apply backward transformations to the points\n
    - points: the points in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - transformations: list of transformations to apply\n
        - The inverse of the transformations are applied in reverse order\n

    Return:\n
    - points_transformed: the transformed points in the form: [[x0, y0, z0], [x1, y1, z1], ...]

    Reference: https://mathematica.stackexchange.com/questions/106257/how-do-i-get-the-inverse-of-a-homogeneous-transformation-matrix
    """
    if len(transformations) == 0:
        return points

    # To homogeneous coordinates
    points_transformed = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # Apply the transformations one by one in reverse order
    for transformation in transformations[::-1]:
        inv_transformation = np.eye(4)
        inv_transformation[:3, :3] = transformation["matrix"][:3, :3].T
        inv_transformation[:3, 3] = -np.matmul(
            transformation["matrix"][:3, :3].T, transformation["matrix"][:3, 3]
        )

        if transformation["type"] == "translation":
            points_transformed = np.matmul(inv_transformation, points_transformed.T).T

        elif transformation["type"] == "rotation":
            axis_origin = np.append(transformation["rotation_axis_origin"], 0)
            points_recentered = points_transformed - axis_origin

            points_rotated = np.matmul(inv_transformation, points_recentered.T).T
            points_transformed = points_rotated + axis_origin

        elif transformation["type"] == "plucker":
            points_transformed = np.matmul(inv_transformation, points_transformed.T).T

        else:
            raise ValueError(f"Unknown transformation type: {transformation['type']}")

    return points_transformed[..., :3]


def _count_points_in_box3d(points, bbox_vertices):
    """
    Count the number of points in a 3D bounding box\n
    - points: the points in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - bbox_vertices: the vertices of the bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
        - The bbox is assumed to be axis-aligned\n

    Return:\n
    - num_points_in_bbox: the number of points in the bounding box
    """

    # Count the number of points in the bounding box
    num_points_in_bbox = np.sum(
        np.all(points >= np.min(bbox_vertices, axis=0), axis=1)
        & np.all(points <= np.max(bbox_vertices, axis=0), axis=1)
    )

    return num_points_in_bbox

def symmetric_chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    Compute symmetric chamfer distance between two point clouds.
    
    Args:
        points1: (N1, 3) point cloud
        points2: (N2, 3) point cloud
    
    Returns:
        Symmetric chamfer distance
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    points1 = torch.from_numpy(points1).cuda()
    points2 = torch.from_numpy(points2).cuda()
    chamfer_forward, _  = chamfer_distance(points1[None, :, :].float(), points2[None, :, :].float())
    chamfer_backward, _ = chamfer_distance(points2[None, :, :].float(), points1[None, :, :].float())

    chamfer_forward = chamfer_forward.cpu().numpy()
    chamfer_backward = chamfer_backward.cpu().numpy()
    # Symmetric chamfer distance
    return (chamfer_forward + chamfer_backward) / 2.0

def sampling_giou(
    bbox1_vertices,
    bbox2_vertices,
    bbox1_transformations,
    bbox2_transformations,
):
    """
    Reference: https://github.com/3dlg-hcvc/singapo/blob/main/metrics/iou_cdist.py

    Compute the IoU between two bounding boxes\n
    - bbox1_vertices: the vertices of the first bounding box\n
    - bbox2_vertices: the vertices of the second bounding box\n
    - bbox1_transformations: list of transformations applied to the first bounding box\n
    - bbox2_transformations: list of transformations applied to the second bounding box\n
    - num_samples (optional): the number of samples to use per bounding box\n

    Return:\n
    - iou: the IoU between the two bounding boxes after applying the transformations
    """
    # if no transformations are applied, use the axis-aligned bounding box IoU
    if len(bbox1_transformations) == 0 and len(bbox2_transformations) == 0:
        return giou_aabb(bbox1_vertices, bbox2_vertices), 0, 0,0,0

    # Volume of the two bounding boxes
    bbox1_volume = np.prod(
        np.max(bbox1_vertices, axis=0) - np.min(bbox1_vertices, axis=0)
    )
    bbox2_volume = np.prod(
        np.max(bbox2_vertices, axis=0) - np.min(bbox2_vertices, axis=0)
    )
    # Volume of the smallest enclosing box
    min_enclosing_bbox = np.minimum(np.min(bbox1_vertices, axis=0), np.min(bbox2_vertices, axis=0))
    max_enclosing_bbox = np.maximum(np.max(bbox1_vertices, axis=0), np.max(bbox2_vertices, axis=0))
    cbbox_volume = np.prod(max_enclosing_bbox - min_enclosing_bbox)

    bbox1_points = _sample_points_in_box3d(bbox1_vertices, num_samples=2048)
    bbox2_points = _sample_points_in_box3d(bbox2_vertices, num_samples=2048)

    bbox1_transformed_vertices = _apply_forward_transformations(bbox1_vertices, bbox1_transformations)
    bbox2_transformed_vertices = _apply_forward_transformations(bbox2_vertices, bbox2_transformations)

    forward_bbox1_points = _apply_forward_transformations(bbox1_points, bbox1_transformations)
    forward_bbox2_points = _apply_forward_transformations(bbox2_points, bbox2_transformations)


    # Transform the forward points to the other box's rest pose frame
    forward_bbox1_points_in_rest_bbox2_frame = _apply_backward_transformations(
        forward_bbox1_points, bbox2_transformations
    )
    forward_bbox2_points_in_rest_bbox1_frame = _apply_backward_transformations(
        forward_bbox2_points, bbox1_transformations
    )

    # Count the number of points in the other bounding box
    num_bbox1_points_in_bbox2 = _count_points_in_box3d(
        forward_bbox1_points_in_rest_bbox2_frame, bbox2_vertices
    )
    num_bbox2_points_in_bbox1 = _count_points_in_box3d(
        forward_bbox2_points_in_rest_bbox1_frame, bbox1_vertices
    )

    # Compute the IoU
    intersect = (
        bbox1_volume * num_bbox1_points_in_bbox2
        + bbox2_volume * num_bbox2_points_in_bbox1
    ) / 2
    union = bbox1_volume * len(bbox1_points) + bbox2_volume * len(bbox2_points) - intersect
    iou = intersect / union

    giou = iou - (cbbox_volume * len(bbox1_points) - union) / (cbbox_volume * len(bbox1_points)) if cbbox_volume > 0 else iou

    return giou, forward_bbox1_points, forward_bbox2_points, bbox1_transformed_vertices, bbox2_transformed_vertices

def compute_per_part_chamfer_distance(
    xyz: np.ndarray,
    xyz_gt: np.ndarray,
    part_ids_pred: np.ndarray,
    part_ids_gt: np.ndarray,
    mapping_pred2gt: Mapping[int, int],
    max_chamfer_distance: float = np.sqrt(3),
) -> Dict[int, float]:
    """
    Compute chamfer distance for each part between prediction and ground truth.
    
    Returns:
        Dictionary mapping part_id to chamfer distance
    """
    # Get unique part IDs from both prediction and ground truth

    part_in_pred = np.unique(part_ids_pred)
    part_in_gt = np.unique(part_ids_gt)
    
    per_part_chamfer_pred_to_gt = {}
    per_part_chamfer_pred_to_gt_nopunish = {}
    for idx, part_id in enumerate(part_in_pred):  
        target_part_idx = int(mapping_pred2gt[idx][0])
        if target_part_idx == -1:
            per_part_chamfer_pred_to_gt[part_id] = max_chamfer_distance
            continue
        target_part_id = int(part_in_gt[target_part_idx])
        pred_points = xyz[part_ids_pred == part_id]
        gt_points = xyz_gt[part_ids_gt == target_part_id]
        chamfer_dist = symmetric_chamfer_distance(pred_points, gt_points)
        per_part_chamfer_pred_to_gt[part_id] = chamfer_dist

        per_part_chamfer_pred_to_gt_nopunish[part_id] = chamfer_dist

    # average chamfer distance
    average_chamfer_pred_to_gt = np.mean(list(per_part_chamfer_pred_to_gt.values()))
    average_chamfer_pred_to_gt_nopunish = np.mean(list(per_part_chamfer_pred_to_gt_nopunish.values()))
    return average_chamfer_pred_to_gt, per_part_chamfer_pred_to_gt, average_chamfer_pred_to_gt_nopunish, per_part_chamfer_pred_to_gt_nopunish

def compute_part_to_all_chamfer_distance(
    xyz_pred: np.ndarray,
    xyz_gt: np.ndarray,
) -> Dict[int, float]:
    """
    Compute chamfer distance for each part to all other parts.
    """

    average_chamfer_articulated = symmetric_chamfer_distance(xyz_pred, xyz_gt)
    return average_chamfer_articulated



def compute_per_part_giou(
    xyz_pred: np.ndarray,
    xyz_gt: np.ndarray,
    part_ids_pred: np.ndarray,
    part_ids_gt: np.ndarray,
    mapping_pred2gt: Mapping[int, int],
    visualize: bool = False,
    draw_bbox: bool = False,

) -> Dict[int, float]:
    # only for articulation state 0/rest state
    per_part_giou = {}
    per_part_giou_nopunish = {}
    unique_gt_part_id = np.unique(part_ids_gt)

    for idx, part_id in enumerate(np.unique(part_ids_pred)):
        target_part_idx = mapping_pred2gt[idx][0]
        if target_part_idx == -1:
            per_part_giou[part_id] = -1.0
            continue
        target_part_id = int(unique_gt_part_id[int(target_part_idx)])
        pred_points = xyz_pred[part_ids_pred == part_id]
        gt_points = xyz_gt[part_ids_gt == target_part_id]
        if len(pred_points) == 0 or len(gt_points) == 0:
            per_part_giou[part_id] = -1.0
            continue
        giou = compute_giou(pred_points, gt_points)
        per_part_giou[part_id] = giou
        per_part_giou_nopunish[part_id] = giou

    average_giou = np.mean(list(per_part_giou.values()))
    average_giou_nopunish = np.mean(list(per_part_giou_nopunish.values()))
    return average_giou, per_part_giou, average_giou_nopunish, per_part_giou_nopunish

def mIoU_aabb(bbox1_vertices, bbox2_verices):
    """
    Compute the generalized IoU between two axis-aligned bounding boxes\n
    - bbox1_vertices: the vertices of the first bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - bbox2_vertices: the vertices of the second bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n

    Return:\n
    - giou: the gIoU between the two bounding boxes
    """
    volume1 = np.prod(np.max(bbox1_vertices, axis=0) - np.min(bbox1_vertices, axis=0))
    volume2 = np.prod(np.max(bbox2_verices, axis=0) - np.min(bbox2_verices, axis=0))

    # Compute the intersection and union of the two bounding boxes
    min_bbox = np.maximum(np.min(bbox1_vertices, axis=0), np.min(bbox2_verices, axis=0))
    max_bbox = np.minimum(np.max(bbox1_vertices, axis=0), np.max(bbox2_verices, axis=0))
    intersection = np.prod(np.clip(max_bbox - min_bbox, a_min=0, a_max=None))
    union = volume1 + volume2 - intersection
    # Compute IoU
    mIoU = intersection / union if union > 0 else 0

    return mIoU

def compute_per_part_mIoU(
    xyz_pred: np.ndarray,
    xyz_gt: np.ndarray,
    part_ids_pred: np.ndarray,
    part_ids_gt: np.ndarray,
    mapping_pred2gt: Mapping[int, int],

) -> Dict[int, float]:
    per_part_mIoU = {}
    per_part_mIoU_nopunish = {}
    unique_gt_part_id = np.unique(part_ids_gt)
    for idx, part_id in enumerate(np.unique(part_ids_pred)):
        target_part_idx = mapping_pred2gt[idx][0]
        if target_part_idx == -1:
            per_part_mIoU[part_id] = 0.0
            continue
        target_part_id = int(unique_gt_part_id[int(target_part_idx)])
        
        pred_bbox_vertices = pcd_to_aabb(xyz_pred[part_ids_pred == part_id])
        gt_bbox_vertices = pcd_to_aabb(xyz_gt[part_ids_gt == target_part_id])
        if len(pred_bbox_vertices) == 0 or len(gt_bbox_vertices) == 0:
            per_part_mIoU[part_id] = 1.0 if np.linalg.norm(pred_bbox_vertices - gt_bbox_vertices) < 0.01 else 0.0
            continue
        mIoU = mIoU_aabb(pred_bbox_vertices, gt_bbox_vertices)
        per_part_mIoU[part_id] = mIoU
        per_part_mIoU_nopunish[part_id] = mIoU
    average_mIoU = np.average(list(per_part_mIoU.values()))
    average_mIoU_nopunish = np.average(list(per_part_mIoU_nopunish.values()))
    return average_mIoU, per_part_mIoU, average_mIoU_nopunish, per_part_mIoU_nopunish


def pcd_to_aabb(
    points: np.ndarray,
) -> np.ndarray:
    """
    Compute 3D AABB for a point cloud.
    """
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    bbox_vertices = np.array([
        [min_point[0], min_point[1], min_point[2]],  # [x0, y0, z0]
        [max_point[0], min_point[1], min_point[2]],  # [x1, y0, z0]
        [min_point[0], max_point[1], min_point[2]],  # [x0, y1, z0]
        [max_point[0], max_point[1], min_point[2]],  # [x1, y1, z0]
        [min_point[0], min_point[1], max_point[2]],  # [x0, y0, z1]
        [max_point[0], min_point[1], max_point[2]],  # [x1, y0, z1]
        [min_point[0], max_point[1], max_point[2]],  # [x0, y1, z1]
        [max_point[0], max_point[1], max_point[2]]   # [x1, y1, z1]
    ])
    return bbox_vertices

def compute_giou(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
) -> float:
    """
    Compute vIoU between two point clouds using oriented bounding boxes.
    Uses the original point cloud points instead of resampling for better accuracy.
    
    Args:
        pred_points: Nx3 array of predicted point cloud
        gt_points: Mx3 array of ground truth point cloud
        
    Returns:
        GIoU value between 0 and 1
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0

    # Compute oriented bounding boxes from point clouds
    pred_bbox_vertices = pcd_to_aabb(pred_points)
    gt_bbox_vertices = pcd_to_aabb(gt_points)

    # Compute volumes
    min_coords1 = np.min(pred_bbox_vertices, axis=0)
    max_coords1 = np.max(pred_bbox_vertices, axis=0)
    aabb_volume1 = np.prod(max_coords1 - min_coords1)
    
    min_coords2 = np.min(gt_bbox_vertices, axis=0)
    max_coords2 = np.max(gt_bbox_vertices, axis=0)
    aabb_volume2 = np.prod(max_coords2 - min_coords2)
    
    volume1 = aabb_volume1
    volume2 = aabb_volume2

    # For intersection, we need to find the actual intersection volume
    min_bbox = np.maximum(min_coords1, min_coords2)
    max_bbox = np.minimum(max_coords1, max_coords2)
    intersection = np.prod(np.clip(max_bbox - min_bbox, a_min=0, a_max=None))
    
    # If there's no AABB intersection, there's no intersection
    if intersection <= 0:
        intersection = 0.0
    
    # Compute union
    union = volume1 + volume2 - intersection
    
    # Compute IoU
    iou = intersection / union if union > 0 else 0
    
    # Compute the smallest enclosing box (still use AABB for this)
    min_enclosing_bbox = np.minimum(min_coords1, min_coords2)
    max_enclosing_bbox = np.maximum(max_coords1, max_coords2)
    enclosing_volume = np.prod(max_enclosing_bbox - min_enclosing_bbox)
    
    # Compute gIoU
    giou = iou - (enclosing_volume - union) / enclosing_volume 
    
    return giou


def giou_aabb(bbox1_vertices, bbox2_vertices):
    """
    Source: https://github.com/3dlg-hcvc/singapo/metrics/giou.py

    Compute the generalized IoU between two axis-aligned bounding boxes
    - bbox1_vertices: the vertices of the first bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]
    - bbox2_vertices: the vertices of the second bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]

    Return:
    - giou: the gIoU between the two bounding boxes
    """
    volume1 = np.prod(np.max(bbox1_vertices, axis=0) - np.min(bbox1_vertices, axis=0))
    volume2 = np.prod(np.max(bbox2_vertices, axis=0) - np.min(bbox2_vertices, axis=0))

    # Compute the intersection and union of the two bounding boxes
    min_bbox = np.maximum(np.min(bbox1_vertices, axis=0), np.min(bbox2_vertices, axis=0))
    max_bbox = np.minimum(np.max(bbox1_vertices, axis=0), np.max(bbox2_vertices, axis=0))
    intersection = np.prod(np.clip(max_bbox - min_bbox, a_min=0, a_max=None))
    union = volume1 + volume2 - intersection
    # Compute IoU
    iou = intersection / union if union > 0 else 0

    # Compute the smallest enclosing box
    min_enclosing_bbox = np.minimum(np.min(bbox1_vertices, axis=0), np.min(bbox2_vertices, axis=0))
    max_enclosing_bbox = np.maximum(np.max(bbox1_vertices, axis=0), np.max(bbox2_vertices, axis=0))
    volume3 = np.prod(max_enclosing_bbox - min_enclosing_bbox)
    
    # Compute gIoU
    giou = iou - (volume3 - union) / volume3 if volume3 > 0 else iou

    if giou < -1:
        print(f"giou is negative: {giou}")

    return giou

def evaluate_articulate_result(
    xyz: np.ndarray,
    xyz_gt: np.ndarray,
    part_ids_pred: np.ndarray,
    part_ids_gt: np.ndarray,
    motion_hierarchy_pred: List[Tuple[int, int]],
    motion_hierarchy_gt: List[Tuple[int, int]],
    is_part_revolute_pred: np.ndarray,
    is_part_revolute_gt: np.ndarray,
    is_part_prismatic_pred: np.ndarray,
    is_part_prismatic_gt: np.ndarray,
    revolute_plucker_pred: np.ndarray,
    revolute_plucker_gt: np.ndarray,
    revolute_range_pred: np.ndarray,
    revolute_range_gt: np.ndarray,
    prismatic_axis_pred: np.ndarray,
    prismatic_axis_gt: np.ndarray,
    prismatic_range_pred: np.ndarray,
    prismatic_range_gt: np.ndarray,
    num_articulation_states: int = 5,
    hungarian_matching_cost_type: str = "cdist",
) -> Dict[str, float]:
    """
    Evaluate articulation results by computing per-part chamfer distance.
    
    Args:
        xyz: (N, 3) point coordinates
        part_ids_pred: (N,) predicted part IDs
        part_ids_gt: (N,) ground truth part IDs
        motion_hierarchy_pred: List of (parent_id, child_id) tuples for prediction
        motion_hierarchy_gt: List of (parent_id, child_id) tuples for ground truth
        is_part_revolute_pred: (num_parts,) boolean array for predicted revolute joints
        is_part_revolute_gt: (num_parts,) boolean array for ground truth revolute joints
        is_part_prismatic_pred: (num_parts,) boolean array for predicted prismatic joints
        is_part_prismatic_gt: (num_parts,) boolean array for ground truth prismatic joints
        revolute_plucker_pred: (num_parts, 6) predicted revolute joint plucker coordinates
        revolute_plucker_gt: (num_parts, 6) ground truth revolute joint plucker coordinates
        revolute_range_pred: (num_parts, 2) predicted revolute joint ranges [low, high]
        revolute_range_gt: (num_parts, 2) ground truth revolute joint ranges [low, high]
        prismatic_axis_pred: (num_parts, 3) predicted prismatic joint axes
        prismatic_axis_gt: (num_parts, 3) ground truth prismatic joint axes
        prismatic_range_pred: (num_parts, 2) predicted prismatic joint ranges [low, high]
        prismatic_range_gt: (num_parts, 2) ground truth prismatic joint ranges [low, high]
        num_articulation_states: Number of articulation states to sample uniformly
    
    Returns:
        Dictionary containing evaluation metrics
    """

    mapping_pred2gt = hungarian_matching_cdist(xyz, part_ids_pred, xyz_gt, part_ids_gt, hungarian_matching_cost_type)
    mapping_gt2pred = hungarian_matching_cdist(xyz_gt, part_ids_gt, xyz, part_ids_pred, hungarian_matching_cost_type)
    
    # find max chamfer in xyz
    bbox_obj = pcd_to_aabb(xyz)
    # find diagonal length of the bbox
    min_corner = bbox_obj.min(axis=0)
    max_corner = bbox_obj.max(axis=0)
    diagonal_length = np.linalg.norm(max_corner - min_corner)
    max_chamfer_distance = diagonal_length/2
        
    # 1. Compute per-part chamfer distance in original pose

    original_per_part_chamfer_pred_to_gt, original_per_part_chanfer, original_per_part_chamfer_pred_to_gt_nopunish, _ = compute_per_part_chamfer_distance(xyz, xyz_gt, part_ids_pred, part_ids_gt, mapping_pred2gt, max_chamfer_distance)
    original_per_part_chamfer_gt_to_pred, _, original_per_part_chamfer_gt_to_pred_nopunish, _ = compute_per_part_chamfer_distance(xyz_gt, xyz, part_ids_gt, part_ids_pred, mapping_gt2pred, max_chamfer_distance)
    original_avg_chamfer = (original_per_part_chamfer_pred_to_gt + original_per_part_chamfer_gt_to_pred) / 2.0
    original_avg_chamfer_nopunish = (original_per_part_chamfer_pred_to_gt_nopunish + original_per_part_chamfer_gt_to_pred_nopunish) / 2.0

    # save xyz points and part ids to file
    overall_original_overall_chamfer = compute_part_to_all_chamfer_distance(xyz, xyz_gt)
    original_per_part_giou_pred_to_gt,  _ , original_per_part_giou_pred_to_gt_nopunish, _ = compute_per_part_giou(xyz, xyz_gt, part_ids_pred, part_ids_gt, mapping_pred2gt)
    original_per_part_giou_gt_to_pred, _ , original_per_part_giou_gt_to_pred_nopunish, _ = compute_per_part_giou(xyz_gt, xyz, part_ids_gt, part_ids_pred, mapping_gt2pred)
    original_avg_giou = (original_per_part_giou_pred_to_gt + original_per_part_giou_gt_to_pred) / 2.0
    original_avg_giou_nopunish = (original_per_part_giou_pred_to_gt_nopunish + original_per_part_giou_gt_to_pred_nopunish) / 2.0
    if original_avg_giou is None or np.isnan(original_avg_giou):
        original_avg_giou = -1.0 

    original_avg_mIoU_pred_to_gt,_ , original_avg_mIoU_pred_to_gt_nopunish, _ = compute_per_part_mIoU(xyz, xyz_gt, part_ids_pred, part_ids_gt, mapping_pred2gt)
    original_avg_mIoU_gt_to_pred,_ , original_avg_mIoU_gt_to_pred_nopunish, _ = compute_per_part_mIoU(xyz_gt, xyz, part_ids_gt, part_ids_pred, mapping_gt2pred)
    original_avg_mIoU = (original_avg_mIoU_pred_to_gt + original_avg_mIoU_gt_to_pred) / 2.0
    original_avg_mIoU_nopunish = (original_avg_mIoU_pred_to_gt_nopunish + original_avg_mIoU_gt_to_pred_nopunish) / 2.0
    if original_avg_mIoU is None or np.isnan(original_avg_mIoU):
        original_avg_mIoU = -1.0

    # 2. Evaluate across different articulation states
    articulated_chamfer_distances = []
    articulated_giou_distances = []
    average_mIoU_all = []

    articulated_chamfer_distances_nopunish = []
    articulated_mIoU_distances_nopunish = []
    articulated_giou_distances_nopunish = []

    test_part_to_all_chamfer_avg = []

    # assume at rest the bbox are aligned with the world axis. 
    aabb_pred_parts_our = np.array([pcd_to_aabb(xyz[part_ids_pred == part_id]) for part_id in np.unique(part_ids_pred)])
    aabb_gt_parts_our = np.array([pcd_to_aabb(xyz_gt[part_ids_gt == part_id]) for part_id in np.unique(part_ids_gt)])

    per_state_chamfer_distances = []
    per_state_giou_distances = []
    per_state_mIoU_distances = []
    per_state_overall_chamfer_distances = []

    per_state_chamfer_distances_nopunish = []
    per_state_giou_distances_nopunish = []
    per_state_mIoU_distances_nopunish = []

    for i in range(num_articulation_states):
        # Sample articulation state uniformly from 0 to 1
        articulation_state = i / max(1, num_articulation_states - 1)
        
        # Articulate prediction
        pred_xyz_articulated = articulate_points(
            xyz, part_ids_pred, motion_hierarchy_pred,
            is_part_revolute_pred, is_part_prismatic_pred,
            revolute_plucker_pred, revolute_range_pred,
            prismatic_axis_pred, prismatic_range_pred,
            articulation_state
        )[0]
        # Articulate ground truth
        gt_xyz_articulated = articulate_points(
            xyz_gt, part_ids_gt, motion_hierarchy_gt,
            is_part_revolute_gt, is_part_prismatic_gt,
            revolute_plucker_gt, revolute_range_gt,
            prismatic_axis_gt, prismatic_range_gt,
            articulation_state
        )[0]
    
        # Compute per-part chamfer distance for this articulation state
        average_chamfer_pred_to_gt, per_part_chamfer_pred_to_gt, average_chamfer_pred_to_gt_nopunish, _ = compute_per_part_chamfer_distance(
            pred_xyz_articulated, gt_xyz_articulated, part_ids_pred, part_ids_gt, mapping_pred2gt, max_chamfer_distance
        )
        average_chamfer_gt_to_pred, per_part_chamfer_gt_to_pred, average_chamfer_gt_to_pred_nopunish, _ = compute_per_part_chamfer_distance(
            gt_xyz_articulated, pred_xyz_articulated, part_ids_gt, part_ids_pred, mapping_gt2pred, max_chamfer_distance
        )

        average_mIoU_pred_to_gt, per_part_mIoU_pred_to_gt, average_mIoU_pred_to_gt_nopunish, _ = compute_per_part_mIoU(
            pred_xyz_articulated, gt_xyz_articulated, part_ids_pred, part_ids_gt, mapping_pred2gt
        )
        average_mIoU_gt_to_pred, per_part_mIoU_gt_to_pred, average_mIoU_gt_to_pred_nopunish, _ = compute_per_part_mIoU(
            gt_xyz_articulated, pred_xyz_articulated, part_ids_gt, part_ids_pred, mapping_gt2pred
        )

        if articulation_state == 0:
            average_giou_pred_to_gt, per_part_giou_pred_to_gt, average_giou_pred_to_gt_nopunish, _ = compute_per_part_giou(
                pred_xyz_articulated, gt_xyz_articulated, part_ids_pred, part_ids_gt, mapping_pred2gt
            )
            average_giou_gt_to_pred, per_part_giou_gt_to_pred, average_giou_gt_to_pred_nopunish, _ = compute_per_part_giou(
                gt_xyz_articulated, pred_xyz_articulated, part_ids_gt, part_ids_pred, mapping_gt2pred
            )
            average_giou = (average_giou_pred_to_gt + average_giou_gt_to_pred) / 2.0
            average_giou_nopunish = (average_giou_pred_to_gt_nopunish + average_giou_gt_to_pred_nopunish) / 2.0
        
        # more accurate way to calculate giou at articulate state 
        else:
            giou_per_part_at_state = []
            giou_per_part_at_state_nopunish = []
            forward_bbox1_points_ls = []
            forward_bbox2_points_ls = []
            bbox1_transformed_vertices_ls = []
            bbox2_transformed_vertices_ls = []
            unique_part_ids_pred = np.unique(part_ids_pred)
            unique_part_ids_gt = np.unique(part_ids_gt)
            bbox_part_id_flatten_our = np.array([np.ones((8,1)) * part_id for part_id in np.unique(part_ids_pred)]).reshape(-1,1)
            bbox_part_id_gt_flatten_our = np.array([np.ones((8,1)) * part_id for part_id in np.unique(part_ids_gt)]).reshape(-1,1)
            bbox_aabb_pred_vertices_flatten_our = aabb_pred_parts_our.reshape(-1, 8, 3)
            bbox_aabb_gt_vertices_flatten_our = aabb_gt_parts_our.reshape(-1, 8, 3)
            part_transformations_pred = articulate_bbox(
            bbox_aabb_pred_vertices_flatten_our, bbox_part_id_flatten_our, motion_hierarchy_pred,
            is_part_revolute_pred, is_part_prismatic_pred,
            revolute_plucker_pred, revolute_range_pred,
            prismatic_axis_pred, prismatic_range_pred,
            articulation_state
            )
            part_transformations_gt = articulate_bbox(
                bbox_aabb_gt_vertices_flatten_our, bbox_part_id_gt_flatten_our, motion_hierarchy_gt,
                is_part_revolute_gt, is_part_prismatic_gt,
                revolute_plucker_gt, revolute_range_gt,
                prismatic_axis_gt, prismatic_range_gt,
                articulation_state
            )
            for idx, part_id in enumerate(unique_part_ids_pred):
                    tgt_part_idx = int(mapping_pred2gt[idx][0])
                    if tgt_part_idx == -1:
                        giou_per_part_at_state.append(-1.0)
                        continue
                    tgt_part_id = unique_part_ids_gt[tgt_part_idx]
                    src_bbox_vertices_our = aabb_pred_parts_our[idx]
                    tgt_bbox_vertices_our = aabb_gt_parts_our[tgt_part_idx]
                    points_src = xyz[part_ids_pred == part_id]
                    points_tgt = xyz_gt[part_ids_gt == tgt_part_id]
                    giou, forward_bbox1_points, forward_bbox2_points, bbox1_transformed_vertices, bbox2_transformed_vertices = sampling_giou(src_bbox_vertices_our, tgt_bbox_vertices_our, part_transformations_pred[idx], part_transformations_gt[tgt_part_idx])
                    forward_bbox1_points_ls.append(forward_bbox1_points)
                    forward_bbox2_points_ls.append(forward_bbox2_points)
                    bbox1_transformed_vertices_ls.append(bbox1_transformed_vertices)
                    bbox2_transformed_vertices_ls.append(bbox2_transformed_vertices)

                    giou_per_part_at_state.append(giou)
                    giou_per_part_at_state_nopunish.append(giou)


            giou_per_part_at_state_forward = np.array(giou_per_part_at_state)
            giou_per_part_at_state_nopunish_forward = np.array(giou_per_part_at_state_nopunish)

            giou_per_part_at_state = []
            giou_per_part_at_state_nopunish = []
            for idx, part_id in enumerate(unique_part_ids_gt):
                src_part_idx = int(mapping_gt2pred[idx][0])
                if src_part_idx == -1:
                    giou_per_part_at_state.append(-1.0)
                    continue
                src_part_id = unique_part_ids_pred[src_part_idx]
                src_bbox_vertices_our = aabb_gt_parts_our[idx]
                tgt_bbox_vertices_our = aabb_pred_parts_our[src_part_idx]
                points_src = xyz_gt[part_ids_gt == part_id]
                points_tgt = xyz[part_ids_pred == src_part_id]
                giou, forward_bbox1_points, forward_bbox2_points, bbox1_transformed_vertices, bbox2_transformed_vertices = sampling_giou(tgt_bbox_vertices_our,src_bbox_vertices_our, part_transformations_gt[idx], part_transformations_pred[src_part_idx])
                giou_per_part_at_state.append(giou)
                giou_per_part_at_state_nopunish.append(giou)
            giou_per_part_at_state_backward = np.array(giou_per_part_at_state)
            giou_per_part_at_state_nopunish_backward = np.array(giou_per_part_at_state_nopunish)

            average_giou_pred_to_gt = np.mean(giou_per_part_at_state_forward)
            average_giou_gt_to_pred = np.mean(giou_per_part_at_state_backward)

            average_giou_nopunish_pred_to_gt = np.mean(giou_per_part_at_state_nopunish_forward)
            average_giou_nopunish_gt_to_pred = np.mean(giou_per_part_at_state_nopunish_backward)

            average_giou = (average_giou_pred_to_gt + average_giou_gt_to_pred) / 2.0
            average_giou_nopunish = (average_giou_nopunish_pred_to_gt + average_giou_nopunish_gt_to_pred) / 2.0
            articulated_giou_distances.append(average_giou)
            articulated_giou_distances_nopunish.append(average_giou_nopunish)

        overall_chamfer_articulated = compute_part_to_all_chamfer_distance(pred_xyz_articulated, gt_xyz_articulated)
        average_chamfer_all = (average_chamfer_pred_to_gt + average_chamfer_gt_to_pred) / 2.0
        average_chamfer_all_nopunish = (average_chamfer_pred_to_gt_nopunish + average_chamfer_gt_to_pred_nopunish) / 2.0
        average_mIoU = (average_mIoU_pred_to_gt + average_mIoU_gt_to_pred) / 2.0
        average_mIoU_nopunish = (average_mIoU_pred_to_gt_nopunish + average_mIoU_gt_to_pred_nopunish) / 2.0
        overall_chamfer_all = overall_chamfer_articulated

        per_state_chamfer_distances.append(average_chamfer_all)
        per_state_giou_distances.append(average_giou)
        per_state_mIoU_distances.append(average_mIoU)
        per_state_overall_chamfer_distances.append(overall_chamfer_all)

        per_state_chamfer_distances_nopunish.append(average_chamfer_all_nopunish)
        per_state_giou_distances_nopunish.append(average_giou_nopunish)
        per_state_mIoU_distances_nopunish.append(average_mIoU_nopunish)

        test_part_to_all_chamfer_avg.append(overall_chamfer_all)
        articulated_chamfer_distances.append(average_chamfer_all)
        articulated_giou_distances.append(average_giou)
        average_mIoU_all.append(average_mIoU)

        articulated_chamfer_distances_nopunish.append(average_chamfer_all_nopunish)
        articulated_giou_distances_nopunish.append(average_giou_nopunish)
        articulated_mIoU_distances_nopunish.append(average_mIoU_nopunish)
        

    # 3. Average across all articulation states
    avg_articulated_chamfer = np.mean(articulated_chamfer_distances)

    avg_articulated_giou = np.mean(articulated_giou_distances)
    avg_overall_chamfer_articulated = np.mean(test_part_to_all_chamfer_avg)
    avg_articulated_mIoU = np.mean(average_mIoU_all)

    avg_articulated_chamfer_nopunish = np.mean(articulated_chamfer_distances_nopunish)
    avg_articulated_giou_nopunish = np.mean(articulated_giou_distances_nopunish)
    avg_articulated_mIoU_nopunish = np.mean(articulated_mIoU_distances_nopunish)

    return {
        'rest_per_part_avg_chamfer': float(original_avg_chamfer),
        'fully_per_part_articulated_avg_chamfer': float(avg_articulated_chamfer),
        'rest_per_part_avg_giou': float(original_avg_giou),
        'fully_per_part_articulated_avg_giou': float(avg_articulated_giou),
        'rest_per_part_avg_mIoU': float(original_avg_mIoU),
        'fully_per_part_articulated_avg_mIoU': float(avg_articulated_mIoU),
        "rest_overall_chamfer": float(overall_original_overall_chamfer), 
        "fully_articulated_overall_chamfer_distances": float(avg_overall_chamfer_articulated),

        "rest_per_part_avg_chamfer_nopunish": float(original_avg_chamfer_nopunish),
        "fully_articulated_avg_chamfer_nopunish": float(avg_articulated_chamfer_nopunish),
        "rest_per_part_avg_giou_nopunish": float(original_avg_giou_nopunish),
        "fully_per_part_articulated_avg_giou_nopunish": float(avg_articulated_giou_nopunish),
        "rest_per_part_avg_mIoU_nopunish": float(original_avg_mIoU_nopunish),
        "fully_per_part_articulated_avg_mIoU_nopunish": float(avg_articulated_mIoU_nopunish),
        
        'per_state_chamfer_distances': [
            float(chamfer) for chamfer in per_state_chamfer_distances
        ],
        'per_state_giou': [
            float(giou) for giou in per_state_giou_distances
        ],
        'per_state_mIoU': [
            float(mIoU) for mIoU in per_state_mIoU_distances
        ],
        'per_state_overall_chamfer_distances': [
            float(overall_chamfer) for overall_chamfer in per_state_overall_chamfer_distances
        ],

        'per_state_chamfer_distances_nopunish': [
            float(chamfer) for chamfer in per_state_chamfer_distances_nopunish
        ],
        'per_state_giou_nopunish': [
            float(giou) for giou in per_state_giou_distances_nopunish
        ],
        'per_state_mIoU_nopunish': [
            float(mIoU) for mIoU in per_state_mIoU_distances_nopunish
        ]
    }, revolute_range_pred, prismatic_range_pred
