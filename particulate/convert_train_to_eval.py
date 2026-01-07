import os
import re
import numpy as np
import glob
from typing import Tuple
import argparse

def get_gt_motion_params(
    link_axes_plucker: np.ndarray, 
    link_range: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract motion parameters from link_axes_plucker and link_range.
    
    Returns:
        gt_part_motion_class: 0="no motion", 1="revolute", 2="prismatic", 3="both"
        gt_revolute_plucker: float32[P, 6]
        gt_prismatic_axis: float32[P, 3]
        gt_revolute_range: float32[P, 2]
        gt_prismatic_range: float32[P, 2]
    """
    # 0: "no motion"; 1: "revolute"; 2: "prismatic"; 3: "both"
    gt_part_motion_class = (
        np.any(link_axes_plucker[:, 6:9] != 0, axis=-1).astype(np.int8) * 2 + \
        np.any(link_axes_plucker[:, 0:3] != 0, axis=-1).astype(np.int8)
    )

    gt_revolute_plucker = link_axes_plucker[:, :6]
    gt_prismatic_axis = link_axes_plucker[:, 6:9]
    gt_revolute_range = link_range[:, :2]
    gt_prismatic_range = link_range[:, 2:]
    return (
        gt_part_motion_class, 
        gt_revolute_plucker, gt_prismatic_axis, 
        gt_revolute_range, gt_prismatic_range
    )

def sample_points_with_part_coverage(
    points: np.ndarray,
    point_to_bone: np.ndarray,
    num_points: int = 100000,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points ensuring at least one point from each unique part_id.
    
    Args:
        points: float32[N, 3] - all available points
        point_to_bone: int32[N] - part_id for each point
        num_points: int - total number of points to sample (default: 100000)
        rng: numpy random generator (optional)
    
    Returns:
        sampled_points: float32[num_points, 3]
        sampled_point_to_bone: int32[num_points]
    """
    rng = np.random.default_rng(seed)
    
    unique_part_ids = np.unique(point_to_bone)
    num_unique_parts = len(unique_part_ids)
    
    # Sample at least one point from each unique part_id
    guaranteed_indices = []
    for part_id in unique_part_ids:
        part_mask = point_to_bone == part_id
        part_indices = np.where(part_mask)[0]
        # Randomly select one point from this part
        selected_idx = rng.choice(part_indices)
        guaranteed_indices.append(selected_idx)
    
    guaranteed_indices = np.array(guaranteed_indices)
    
    # Sample remaining points uniformly from all points
    remaining_needed = num_points - num_unique_parts
    all_indices = np.arange(len(points))
    available_indices = np.setdiff1d(all_indices, guaranteed_indices)
    
    if remaining_needed > 0:
        additional_indices = rng.choice(available_indices, size=remaining_needed, replace=False)
        sampled_indices = np.concatenate([guaranteed_indices, additional_indices])
    else:
        sampled_indices = guaranteed_indices
    
    # Extract sampled points and their part_ids
    sampled_points = points[sampled_indices].astype(np.float32)
    sampled_point_to_bone = point_to_bone[sampled_indices].astype(np.int32)
    
    return sampled_points, sampled_point_to_bone

def convert_to_eval_format(input_file: str, output_file: str, num_points: int = 100000, seed: int = 0):
    """
    Convert training format to evaluation format.
    """
    # Load input data
    data = np.load(input_file)
    
    # Extract required fields
    points = data["points"].astype(np.float32)
    point_to_bone = data["point_to_bone"].astype(np.int32)
    bone_structure = data["bone_structure"]
    link_axes_plucker = data["link_axes_plucker"].astype(np.float32)
    link_range = data["link_range"].astype(np.float32)
    
    # Sample 100k points ensuring at least one from each unique part_id
    if num_points > len(points):
        print(f"Warning: num_points ({num_points}) is greater than the number of points ({len(points)}). Using all points.")
        sampled_points = points
        sampled_point_to_bone = point_to_bone
    else:
        sampled_points, sampled_point_to_bone = sample_points_with_part_coverage(
            points, point_to_bone, num_points=num_points, seed=seed
        )
    
    # Convert bone_structure to list of tuples
    if isinstance(bone_structure, np.ndarray):
        if bone_structure.ndim == 2 and bone_structure.shape[1] == 2:
            motion_hierarchy = [tuple(int(x) for x in row) for row in bone_structure]
        else:
            raise ValueError(f"Unexpected bone_structure shape: {bone_structure.shape}")
    else:
        motion_hierarchy = bone_structure
    
    # Compute motion parameters
    (
        gt_part_motion_class, 
        gt_revolute_plucker, gt_prismatic_axis, 
        gt_revolute_range, gt_prismatic_range
    ) = get_gt_motion_params(link_axes_plucker, link_range)
    
    is_part_revolute = (gt_part_motion_class == 1) | (gt_part_motion_class == 3)
    is_part_prismatic = (gt_part_motion_class == 2) | (gt_part_motion_class == 3)
    
    # Save in evaluation format
    np.savez(
        output_file,
        points=sampled_points,
        part_ids=sampled_point_to_bone,
        motion_hierarchy=motion_hierarchy,
        is_part_revolute=is_part_revolute,
        is_part_prismatic=is_part_prismatic,
        revolute_plucker=gt_revolute_plucker,
        revolute_range=gt_revolute_range,
        prismatic_axis=gt_prismatic_axis,
        prismatic_range=gt_prismatic_range,
    )
    print(f"Converted: {input_file} -> {output_file} (sampled {len(sampled_points)} points from {len(points)} original)")

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--num_points", type=int, default=100000, help="Number of points to sample (default: 100000)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    return parser.parse_args()

def main():
    args = argparser()
    # Configuration
    input_file = args.input_file
    output_file = args.output_file
            
    # Convert
    convert_to_eval_format(input_file, output_file, num_points=args.num_points, seed=args.seed)

if __name__ == "__main__":
    main()
