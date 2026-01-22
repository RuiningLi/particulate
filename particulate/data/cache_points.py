import argparse
import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
import trimesh

from particulate.data_utils import (
    load_obj_raw_preserve,
    sharp_sample_pointcloud,
    get_face_to_bone_mapping,
    get_gt_motion_params,
    AXES_PLUCKER_DIM,
    RANGE_DIM
)


def cache_points(
    root: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    num_points: int = 8192,
    ratio_sharp: float = 0.5,
    format: str = "train",
) -> bool:
    """
    Caches point features for a given list of render paths.
    """
    if output_path is None:
        output_path = Path(root) / "points.npz"
    else:
        output_path = Path(output_path)
    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)
    
    verts, faces = load_obj_raw_preserve(Path(root) / "original.obj")
    mesh = trimesh.Trimesh(verts, faces, process=False)

    num_points_sharp_edges = int(num_points * ratio_sharp)
    num_points_uniform = num_points - num_points_sharp_edges
    points_sharp, normals_sharp, _, _, vertex_ids_a, vertex_ids_b = sharp_sample_pointcloud(mesh, num_points_sharp_edges)

    # Get the point to bone mapping
    meta = np.load(Path(root) / "meta.npz")
    vert_to_bone = meta["vert_to_bone"]
    
    # If no sharp edges were found, sample all points uniformly
    if len(points_sharp) == 0 and ratio_sharp > 0:
        print(f"Warning: No sharp edges found in {root}, sampling all points uniformly")
        num_points_uniform = num_points
        point_to_bone_sharp = np.array([], dtype=np.int32)
    else:
        assert np.all(vert_to_bone[vertex_ids_a] == vert_to_bone[vertex_ids_b])
        point_to_bone_sharp = vert_to_bone[vertex_ids_a]

    # Sample points on the surface with normals
    points_uniform, face_indices = mesh.sample(num_points_uniform, return_index=True)
    normals_uniform = mesh.face_normals[face_indices]
    face_to_bone = get_face_to_bone_mapping(vert_to_bone, faces)
    if face_to_bone is None:
        print("Warning: Some faces do not have all vertices belonging to the same bone")
        return False
    point_to_bone_uniform = face_to_bone[face_indices]

    points = np.concatenate([points_sharp, points_uniform], axis=0)
    normals = np.concatenate([normals_sharp, normals_uniform], axis=0)
    point_to_bone = np.concatenate([point_to_bone_sharp, point_to_bone_uniform], axis=0)
    point_from_sharp = np.concatenate([
        np.ones(len(points_sharp), dtype=np.bool_),
        np.zeros(len(points_uniform), dtype=np.bool_)
    ], axis=0)

    # load Meta data and plucker coordinates
    meta = np.load(Path(root) / "meta.npz")
    bone_structure = meta["bone_structure"]
    num_bones = meta["vert_to_bone"].max() + 1

    # get plucker coordinates
    link_axes_plucker_path = Path(root) / "link_axes_plucker.npz"
    link_range_path = Path(root) / "link_range.npz"
    link_axes_plucker = np.load(link_axes_plucker_path)
    link_range = np.load(link_range_path)

    combined_link_axes_plucker = np.zeros((num_bones, AXES_PLUCKER_DIM), dtype=np.float32)
    combined_link_range = np.zeros((num_bones, RANGE_DIM), dtype=np.float32)
    for k, v in link_axes_plucker.items():
        combined_link_axes_plucker[int(k)] = v
    for k, v in link_range.items():
        combined_link_range[int(k)] = v

    if format == "train":
        np.savez(
            output_path,
            points=points,
            normals=normals,
            point_to_bone=point_to_bone,
            point_from_sharp=point_from_sharp,
            bone_structure=bone_structure,
            link_axes_plucker=combined_link_axes_plucker,
            link_range=combined_link_range,
        )
    elif format == "eval":
        if isinstance(bone_structure, np.ndarray):
            assert bone_structure.ndim == 2 and bone_structure.shape[1] == 2
            motion_hierarchy = [tuple(int(x) for x in row) for row in bone_structure]
        else:
            motion_hierarchy = bone_structure

        (
            gt_part_motion_class, 
            gt_revolute_plucker, gt_prismatic_axis, 
            gt_revolute_range, gt_prismatic_range
        ) = get_gt_motion_params(combined_link_axes_plucker, combined_link_range)

        is_part_revolute = (gt_part_motion_class == 1) | (gt_part_motion_class == 3)
        is_part_prismatic = (gt_part_motion_class == 2) | (gt_part_motion_class == 3)

        np.savez(
            output_path,
            points=points,
            part_ids=point_to_bone,
            motion_hierarchy=motion_hierarchy,
            is_part_revolute=is_part_revolute,
            is_part_prismatic=is_part_prismatic,
            revolute_plucker=gt_revolute_plucker,
            revolute_range=gt_revolute_range,
            prismatic_axis=gt_prismatic_axis,
            prismatic_range=gt_prismatic_range,
        )
    else:
        raise ValueError(f"Invalid format: {format}")

    return True

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_points", type=int, default=50_000)
    parser.add_argument("--ratio_sharp", type=float, default=0.5)
    parser.add_argument("--format", type=str, default="train", choices=["train", "eval"])
    args = parser.parse_args()

    if args.format == "eval" and args.ratio_sharp > 0:
        print("Warning: ratio_sharp is ignored for eval format as we uniformly sample points for Chamfer Distance computation during evaluation")
        args.ratio_sharp = 0

    cache_points(
        Path(args.root),
        Path(args.output_path),
        num_points=args.num_points,
        ratio_sharp=args.ratio_sharp,
        format=args.format,
    )
