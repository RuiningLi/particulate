import argparse
import os
from pathlib import Path
import glob
import numpy as np
import trimesh

from particulate.data_utils import (
    load_obj_raw_preserve,
    get_gt_motion_params,
    get_face_to_bone_mapping,
    AXES_PLUCKER_DIM,
    RANGE_DIM
)


def cache_points(
    root: str,
    output_path: str,
    num_points: int = 8192,
) -> bool:
    """
    Caches point features for a given list of render paths.
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    os.makedirs(output_dir, exist_ok=True)
    
    verts, faces = load_obj_raw_preserve(Path(root))
    mesh = trimesh.Trimesh(verts, faces, process=False)

    # Get the point to bone mapping
    meta_path = root.replace("original.obj", "meta.npz")
    meta = np.load(meta_path)
    vert_to_bone = meta["vert_to_bone"]

    # Sample points on the surface with normals
    points, face_indices = mesh.sample(num_points, return_index=True)
    normals = mesh.face_normals[face_indices]
    face_to_bone = get_face_to_bone_mapping(vert_to_bone, faces)
    if face_to_bone is None:
        print("Warning: Some faces do not have all vertices belonging to the same bone")
        return False
    point_to_bone = face_to_bone[face_indices]

    # load plucker coordinates
    bone_structure = meta["bone_structure"]
    num_bones = meta["vert_to_bone"].max() + 1

    # get plucker coordinates
    link_axes_plucker_path = root.replace("original.obj", "link_axes_plucker.npz")
    link_range_path = root.replace("original.obj", "link_range.npz")
    link_axes_plucker = np.load(link_axes_plucker_path)
    link_range = np.load(link_range_path)

    combined_link_axes_plucker = np.zeros((num_bones, AXES_PLUCKER_DIM), dtype=np.float32)
    combined_link_range = np.zeros((num_bones, RANGE_DIM), dtype=np.float32)
    for k, v in link_axes_plucker.items():
        combined_link_axes_plucker[int(k)] = v
    for k, v in link_range.items():
        combined_link_range[int(k)] = v
    
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
    ) = get_gt_motion_params(combined_link_axes_plucker, combined_link_range)
    
    is_part_revolute = (gt_part_motion_class == 1) | (gt_part_motion_class == 3)
    is_part_prismatic = (gt_part_motion_class == 2) | (gt_part_motion_class == 3)
    
    # Save in evaluation format
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
    return True

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_points", type=int, default=100_000)
    args = parser.parse_args()

    obj_files = glob.glob(os.path.join(args.root_dir, "**", "original.obj"))
    for obj_file in obj_files:
        obj_name = obj_file.split("/")[-2]
        output_path = os.path.join(args.output_dir, f"{obj_name}.npz")
        if not os.path.exists(output_path):
            try:
                cache_points(
                    obj_file,
                    output_path,
                    num_points=args.num_points,
                )
            except Exception as e:
                print(f"Error caching points for {obj_file}: {e}")
                continue
