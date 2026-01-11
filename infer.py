import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import trimesh
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from particulate.articulation_utils import plucker_to_axis_point
from particulate.data_utils import sample_points, load_obj_raw_preserve
from particulate.export_utils import export_animated_glb_file, export_urdf, export_mjcf
from particulate.models import *
from particulate.postprocessing_utils import find_part_ids_for_faces
from particulate.visualization_utils import (
    ARROW_COLOR_PRISMATIC,
    ARROW_COLOR_REVOLUTE,
    create_arrow,
    create_ring,
    create_textured_mesh_parts,
    get_3D_arrow_on_points,
)
from partfield_utils import get_partfield_model, obtain_partfield_feats

from yacs.config import CfgNode
torch.serialization.add_safe_globals([CfgNode])


DATA_CONFIG = {
    'sharp_point_ratio': 0.5,
    'normalize_points': True
}


def prepare_inputs(mesh, num_points_global: int = 40000, num_points_decode: int = 2048, device: str = "cuda"):
    """Prepare inputs from a mesh file for model inference."""
    sharp_point_ratio = DATA_CONFIG['sharp_point_ratio']
    all_points, _, _, _ = sample_points(mesh, num_points_global, sharp_point_ratio)
    points, normals, sharp_flag, face_indices = sample_points(mesh, num_points_decode, sharp_point_ratio, at_least_one_point_per_face=True)

    if DATA_CONFIG['normalize_points']:
        bbmin = np.concatenate([all_points, points], axis=0).min(0)
        bbmax = np.concatenate([all_points, points], axis=0).max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 1.0 / (bbmax - bbmin).max()
        all_points = (all_points - center) * scale
        points = (points - center) * scale

    all_points = torch.from_numpy(all_points).to(device).float().unsqueeze(0)
    points = torch.from_numpy(points).to(device).float().unsqueeze(0)
    normals = torch.from_numpy(normals).to(device).float().unsqueeze(0)
    
    partfield_model = get_partfield_model(device=device)
    feats = obtain_partfield_feats(partfield_model, all_points, points)

    return dict(xyz=points, normals=normals, feats=feats), sharp_flag, face_indices


@torch.no_grad()
def infer_single_asset(
    mesh,
    up_dir,
    model,
    num_points,
    min_part_confidence=0.0
):
    mesh_transformed = mesh.copy()
    if up_dir == "X":
        rotation_matrix = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
        mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
    elif up_dir == "-X":
        rotation_matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
        mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
    elif up_dir == "Y":
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
    elif up_dir == "-Y":
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
    elif up_dir == "Z":
        pass
    elif up_dir == "-Z":
        rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
    else:
        raise ValueError(f"Invalid up direction: {up_dir}")

    # Normalize mesh to [-0.5, 0.5]^3 bounding box
    bbox_min = mesh_transformed.vertices.min(axis=0)
    bbox_max = mesh_transformed.vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_transformed.vertices -= center
    scale = (bbox_max - bbox_min).max()
    mesh_transformed.vertices /= scale

    inputs, sharp_flag, face_indices = prepare_inputs(mesh_transformed, num_points_global=40000, num_points_decode=num_points)

    with torch.no_grad():
        outputs = model.infer(
            xyz=inputs['xyz'],
            feats=inputs['feats'],
            normals=inputs['normals'],
            output_all_hyps=True,
            min_part_confidence=min_part_confidence
        )
    
    return outputs, face_indices, mesh_transformed


def save_articulated_meshes(mesh, face_indices, outputs, output_path, strict, animation_frames: int = 50, hyp_idx: int = 0, save_name: str = None):
    part_ids = outputs[hyp_idx]['part_ids']
    motion_hierarchy = outputs[hyp_idx]['motion_hierarchy']
    is_part_revolute = outputs[hyp_idx]['is_part_revolute']
    is_part_prismatic = outputs[hyp_idx]['is_part_prismatic']
    revolute_plucker = outputs[hyp_idx]['revolute_plucker']
    revolute_range = outputs[hyp_idx]['revolute_range']
    prismatic_axis = outputs[hyp_idx]['prismatic_axis']
    prismatic_range = outputs[hyp_idx]['prismatic_range']

    face_part_ids = find_part_ids_for_faces(mesh, part_ids, face_indices, strict=strict)
    unique_part_ids = np.unique(face_part_ids)
    num_parts = len(unique_part_ids)
    print(f"Found {num_parts} unique parts")
    
    mesh_parts_original = [mesh.submesh([face_part_ids == part_id], append=True) for part_id in unique_part_ids]
    mesh_parts_segmented = create_textured_mesh_parts([mp.copy() for mp in mesh_parts_original])

    # Create axes
    axes = []
    for i, mesh_part in enumerate(mesh_parts_segmented):
        part_id = unique_part_ids[i]
        if is_part_revolute[part_id]:
            axis, point = plucker_to_axis_point(revolute_plucker[part_id])
            arrow_start, arrow_end = get_3D_arrow_on_points(axis, mesh_part.vertices, fixed_point=point, extension=0.2)
            axes.append(create_arrow(arrow_start, arrow_end, color=ARROW_COLOR_REVOLUTE, radius=0.01, radius_tip=0.018))
            # Add rings at arrow_start and arrow_end
            arrow_dir = arrow_end - arrow_start
            axes.append(create_ring(arrow_start, arrow_dir, major_radius=0.03, minor_radius=0.006, color=ARROW_COLOR_REVOLUTE))
            axes.append(create_ring(arrow_end, arrow_dir, major_radius=0.03, minor_radius=0.006, color=ARROW_COLOR_REVOLUTE))
        elif is_part_prismatic[part_id]:
            axis = prismatic_axis[part_id]
            arrow_start, arrow_end = get_3D_arrow_on_points(axis, mesh_part.vertices, extension=0.2)
            axes.append(create_arrow(arrow_start, arrow_end, color=ARROW_COLOR_PRISMATIC, radius=0.01, radius_tip=0.018))

    if save_name is None:
        mesh_parts_filename = "mesh_parts_with_axes.glb"
        animated_filename = "animated_textured.glb"
    else:
        mesh_parts_filename = f"mesh_parts_with_axes_{save_name}.glb"
        animated_filename = f"animated_textured_{save_name}.glb"
    
    trimesh.Scene(mesh_parts_segmented + axes).export(Path(output_path) / mesh_parts_filename)
    print(f"Saved prediction GLB to {os.path.join(output_path, mesh_parts_filename)}")

    print("Exporting animated GLB files...")
        
    export_animated_glb_file(
        mesh_parts_original,
        unique_part_ids,
        motion_hierarchy,
        is_part_revolute,
        is_part_prismatic,
        revolute_plucker,
        revolute_range,
        prismatic_axis,
        prismatic_range,
        animation_frames,
        str(Path(output_path) / animated_filename),
        include_axes=False,
        axes_meshes=None
    )
    print(f"Saved animated GLB to {os.path.join(output_path, animated_filename)}")

    return (
        mesh_parts_original,
        face_part_ids,
        unique_part_ids,
        motion_hierarchy,
        is_part_revolute,
        is_part_prismatic,
        revolute_plucker,
        revolute_range,
        prismatic_axis,
        prismatic_range
    )


def main(args):
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configuration
    print(f"Loading model config from {args.model_config}")
    cfg = OmegaConf.load(args.model_config)

    # Initialize model
    print("Initializing model...")
    model_size = cfg.get('model_size', 'B')
    cfg.pop('model_size', None)
    model = eval(f"PAT_{model_size}")(**cfg)
    model.eval()
    
    # Load weights
    print("Downloading/Loading model from Hugging Face...")
    model_checkpoint = hf_hub_download(repo_id="rayli/Particulate", filename=f"model.pt")
    model.load_state_dict(torch.load(model_checkpoint, map_location="cpu"))
    model.to("cuda")
    
    # Download PartField model if needed
    partfield_model_dir = os.path.join("PartField", "model")
    os.makedirs(partfield_model_dir, exist_ok=True)
    hf_hub_download(repo_id="mikaelaangel/partfield-ckpt", filename="model_objaverse.ckpt", local_dir=partfield_model_dir)
    print("Models loaded successfully.")
    
    # Load mesh
    print(f"Loading mesh from {args.input_mesh}")
    if args.input_mesh.endswith(".obj"):
        verts, faces = load_obj_raw_preserve(Path(args.input_mesh))
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    else:
        mesh = trimesh.load(args.input_mesh, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
            
    # Run inference
    print("Running inference...")
    outputs, face_indices, mesh_transformed = infer_single_asset(
        mesh=mesh,
        up_dir=args.up_dir,
        model=model,
        num_points=args.num_points,
        min_part_confidence=args.min_part_confidence
    )
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strict = not args.no_strict
    (
        mesh_parts_original,
        face_part_ids,
        unique_part_ids,
        motion_hierarchy,
        is_part_revolute,
        is_part_prismatic,
        revolute_plucker,
        revolute_range,
        prismatic_axis,
        prismatic_range
    ) = save_articulated_meshes(
        mesh_transformed, face_indices, outputs,
        output_path=args.output_dir,
        strict=strict,
        animation_frames=args.animation_frames,
        save_name=timestamp
    )
        
    # Export URDF
    if args.export_urdf:
        urdf_output_path = os.path.join(args.output_dir, f"urdf_{timestamp}", "model.urdf")
        export_urdf(
            mesh_parts_original,
            unique_part_ids,
            motion_hierarchy,
            is_part_revolute,
            is_part_prismatic,
            revolute_plucker,
            revolute_range,
            prismatic_axis,
            prismatic_range,
            output_path=urdf_output_path,
            name="model"
        )
        
    # Export MJCF
    if args.export_mjcf:
        mjcf_output_path = os.path.join(args.output_dir, f"mjcf_{timestamp}", "model.xml")
        export_mjcf(
            mesh_parts_original,
            unique_part_ids,
            motion_hierarchy,
            is_part_revolute,
            is_part_prismatic,
            revolute_plucker,
            revolute_range,
            prismatic_axis,
            prismatic_range,
            output_path=mjcf_output_path,
            name="model"
        )

    # Save results for evaluation
    if args.eval:
        eval_result_output_dir = os.path.join(args.output_dir, "eval")
        os.makedirs(eval_result_output_dir, exist_ok=True)
        mesh.export(os.path.join(eval_result_output_dir, "pred.obj"))

        old_part_id_to_new_part_id = {part_id: idx for idx, part_id in enumerate(unique_part_ids)}
        new_face_part_ids = face_part_ids.copy()
        for idx, part_id in enumerate(unique_part_ids):
            new_face_part_ids[face_part_ids == part_id] = idx
        new_motion_hierarchy = [
            (old_part_id_to_new_part_id[p], old_part_id_to_new_part_id[c]) for p, c in motion_hierarchy
        ]

        np.savez(os.path.join(eval_result_output_dir, "pred.npz"),
            face_part_ids=new_face_part_ids,
            motion_hierarchy=new_motion_hierarchy,
            is_part_revolute=is_part_revolute[unique_part_ids],
            is_part_prismatic=is_part_prismatic[unique_part_ids],
            revolute_plucker=revolute_plucker[unique_part_ids],
            revolute_range=revolute_range[unique_part_ids],
            prismatic_axis=prismatic_axis[unique_part_ids],
            prismatic_range=prismatic_range[unique_part_ids],
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")

    parser = argparse.ArgumentParser(description="Particulate Inference Script")
    parser.add_argument("--input_mesh", type=str, required=True, help="Path to input mesh (.obj or .glb)")
    parser.add_argument("--output_dir", type=str, default="inference_outputs", help="Directory to save outputs")
    parser.add_argument("--model_config", type=str, default="configs/particulate-B.yaml", help="Path to model config")
    parser.add_argument("--up_dir", type=str, default="-Z", choices=["X", "Y", "Z", "-X", "-Y", "-Z"], help="Up direction of the input mesh")
    parser.add_argument("--num_points", type=int, default=102400, help="Number of points to sample")
    parser.add_argument("--min_part_confidence", type=float, default=0.0, help="Minimum part confidence")
    parser.add_argument("--no_strict", action="store_true", help="Disable strict connected component refinement")
    parser.add_argument("--animation_frames", type=int, default=50, help="Number of animation frames")
    parser.add_argument("--export_urdf", action="store_true", help="Export URDF")
    parser.add_argument("--export_mjcf", action="store_true", help="Export MJCF")
    parser.add_argument("--eval", action="store_true", help="Save results for evaluation")
    args = parser.parse_args()
    main(args)
