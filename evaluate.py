import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
import trimesh
from particulate.data_utils import load_obj_raw_preserve, get_face_to_bone_mapping, get_gt_motion_params
from particulate.evaluation_utils import evaluate_articulate_result

import numpy as np
import torch
from tqdm import tqdm


import sys

@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def evaluate_inference_results(
    results: Dict[str, Any],
    gt: Dict[str, Any],
    output_path: str,
    hungarian_matching_cost_type = "cdist", # "cdist"or "chamfer"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    """Evaluate inference results."""
    eval_results, revolute_range_pred, prismatic_range_pred = evaluate_articulate_result(
        xyz=results['xyz'],
        xyz_gt=gt['xyz'],
        part_ids_pred=results['part_ids'],
        part_ids_gt=gt['part_ids'],
        motion_hierarchy_pred=results['motion_hierarchy'],
        motion_hierarchy_gt=gt['motion_hierarchy'],
        is_part_revolute_pred=results['is_part_revolute'],
        is_part_revolute_gt=gt['is_part_revolute'],
        is_part_prismatic_pred=results['is_part_prismatic'],
        is_part_prismatic_gt=gt['is_part_prismatic'],
        revolute_plucker_pred=results['revolute_plucker'],
        revolute_plucker_gt=gt['revolute_plucker'],
        revolute_range_pred=results['revolute_range'],
        revolute_range_gt=gt['revolute_range'],
        prismatic_axis_pred=results['prismatic_axis'],
        prismatic_axis_gt=gt['prismatic_axis'],
        prismatic_range_pred=results['prismatic_range'],
        prismatic_range_gt=gt['prismatic_range'],
        num_articulation_states=5,
        hungarian_matching_cost_type = hungarian_matching_cost_type,
    )
    json.dump(eval_results, open(output_path.parent / f"{output_path.stem}_eval.json", "w"), indent=4)
    return eval_results, revolute_range_pred, prismatic_range_pred

def process_custom_prediction(
    meta_dir: str,
    num_points: int,
    save_dir: str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    obj_files = glob.glob(os.path.join(meta_dir, "**", "original.obj"))
    for obj_file in obj_files:
        category_name = obj_file.split("/")[-2]
        if os.path.exists(os.path.join(save_dir, f"{category_name}.npz")):
            continue
        meta_path = obj_file.replace("original.obj", "meta.npz")
        meta = np.load(meta_path)
        face_part_id = meta["vert_to_bone"]
        verts, faces = load_obj_raw_preserve(Path(obj_file))
        mesh = trimesh.Trimesh(verts, faces, process=False)
        # Sample points on the surface with normals
        points_uniform, face_indices = mesh.sample(num_points, return_index=True)
        face_to_bone = get_face_to_bone_mapping(face_part_id, faces)
        point_to_bone_uniform = face_to_bone[face_indices]
        np.savez(os.path.join(save_dir, f"{category_name}.npz"),
            points=points_uniform,
            point_to_bone=point_to_bone_uniform,
            motion_hierarchy=meta['motion_hierarchy'],
            is_part_revolute=meta['is_part_revolute'],
            is_part_prismatic=meta['is_part_prismatic'],
            revolute_plucker=meta['revolute_plucker'],
            revolute_range=meta['revolute_range'],
            prismatic_axis=meta['prismatic_axis'],
            prismatic_range=meta['prismatic_range'],
            face_indices=face_indices,
        )
    print(f"Processed {len(obj_files)} objects")

def main(
    gt_dir: str,
    result_dir: str,
    output_dir: str = "./inference_results",
    device: str = "cuda",
    save_pcd: bool = True,
    save_pcd_gt: bool = False,
    result_type: str = "particulate",
    **kwargs
):
    """Main inference function."""
    # Validate device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    eval_results = []
    if result_type == "particulate":
        pred_files = glob.glob(os.path.join(result_dir, "*.npz"))
    elif result_type == "custom":
        # run the resample_points_release.py to resample points on the mesh
        pred_files = glob.glob(os.path.join(result_dir, "*.npz"))

    num_failed = 0
    for i, pred_file in enumerate(tqdm(pred_files, desc="Processing samples")):
        # Get sample

        if result_type == "custom":
            sample_name = pred_file.split("/")[-1].split(".")[0]
            model_name = sample_name
            try:
                gt_file = os.path.join(gt_dir, f"{model_name}.npz")
                gt_sample = np.load(gt_file)
                pred = np.load(pred_file)
            except:
                continue
            results = {
                'xyz': pred['points'],
                'part_ids': pred['point_to_bone'],
                'motion_hierarchy': pred['motion_hierarchy'],
                'is_part_revolute': pred['is_part_revolute'],
                'is_part_prismatic': pred['is_part_prismatic'],
                'revolute_plucker': pred['revolute_plucker'],
                'revolute_range': pred['revolute_range'],
                'prismatic_axis': pred['prismatic_axis'],
                'prismatic_range': pred['prismatic_range'],
            }

        elif result_type == "particulate":
            sample_name = pred_file.split("/")[-1]
            model_name = sample_name.split(".")[0]
            try:
                gt_file = os.path.join(gt_dir, f"{model_name}.npz")
                gt_sample = np.load(gt_file)
                pred = np.load(pred_file)
            except:
                continue
            results = {
                'xyz': pred['points'],
                'part_ids': pred['part_ids'],
                'motion_hierarchy': pred['motion_hierarchy'],
                'is_part_revolute': pred['is_part_revolute'],
                'is_part_prismatic': pred['is_part_prismatic'],
                'revolute_plucker': pred['revolute_plucker'],
                'revolute_range': pred['revolute_range'],
                'prismatic_axis': pred['prismatic_axis'],
                'prismatic_range': pred['prismatic_range'],
            }
        else:
            raise ValueError(f"Invalid result type: {result_type}")

        gt = {
            'part_ids': gt_sample['part_ids'],
            'xyz': gt_sample['points'],
            'motion_hierarchy': gt_sample['motion_hierarchy'],
            'is_part_revolute': gt_sample['is_part_revolute'],
            'is_part_prismatic':gt_sample['is_part_prismatic'],
            'revolute_plucker': gt_sample['revolute_plucker'],
            'revolute_range': gt_sample['revolute_range'],
            'prismatic_axis': gt_sample['prismatic_axis'],
            'prismatic_range': gt_sample['prismatic_range'],
        }


        # Save results
        output_file = output_path / f"{sample_name}_pred"

        if os.path.exists(os.path.join(output_file.parent / f"{output_file.stem}_eval.json")):
            eval_result = json.load(open(os.path.join(output_file.parent / f"{output_file.stem}_eval.json"), "r"))
            eval_results.append(eval_result)
        else:
            eval_result, revolute_range_pred, prismatic_range_pred = evaluate_inference_results(results, gt, str(output_file), hungarian_matching_cost_type='cdist')
            eval_results.append(eval_result)

        if os.path.exists(os.path.join(output_file.parent / f"{output_file.stem}_eval.json")):
            eval_result = json.load(open(os.path.join(output_file.parent / f"{output_file.stem}_eval.json"), "r"))
            eval_results.append(eval_result)
            continue



    overall_eval_results = {
        'rest_avg_chamfer': np.round(np.mean([result['rest_avg_chamfer'] for result in eval_results]), 4),
        'rest_avg_giou': np.round(np.mean([result['rest_avg_giou'] for result in eval_results]), 4),
        'rest_avg_mIoU': np.round(np.mean([result['rest_avg_mIoU'] for result in eval_results]), 4),
    }
    overall_eval_results['fully_articulated_avg_chamfer'] = np.round(np.mean([result['per_state_chamfer_distances'][-1] for result in eval_results]), 4)
    overall_eval_results['fully_articulated_avg_giou'] = np.round(np.mean([result['per_state_giou'][-1] for result in eval_results]), 4)
    overall_eval_results['fully_articulated_overall_chamfer_distances'] = np.round(np.mean([result['per_state_overall_chamfer_distances'][-1] for result in eval_results]), 4)
    json.dump(overall_eval_results, open(output_path.parent / f"{output_path.stem}_eval_overall.json", "w"), indent=4)

    overall_eval_results_nopunish = {
        'rest_avg_chamfer_nopunish': np.round(np.mean([result['rest_avg_chamfer_nopunish'] for result in eval_results]), 4),
        'rest_avg_giou_nopunish': np.round(np.mean([result['rest_avg_giou_nopunish'] for result in eval_results]), 4),
        'rest_avg_mIoU_nopunish': np.round(np.mean([result['rest_avg_mIoU_nopunish'] for result in eval_results]), 4),
    }
    
    overall_eval_results_nopunish['fully_articulated_avg_chamfer_nopunish'] = np.round(np.mean([result['per_state_chamfer_distances_nopunish'][-1] for result in eval_results]), 4)
    overall_eval_results_nopunish['fully_articulated_avg_giou_nopunish'] = np.round(np.mean([result['per_state_giou_nopunish'][-1] for result in eval_results]), 4)
    overall_eval_results_nopunish['fully_articulated_overall_chamfer_distances_nopunish'] = np.round(np.mean([result['per_state_overall_chamfer_distances'][-1] for result in eval_results]), 4)
    json.dump(overall_eval_results_nopunish, open(output_path.parent / f"{output_path.stem}_eval_overall_nopunish.json", "w"), indent=4)
    print(f"Inference completed! Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on Articulate3D model")
    parser.add_argument("--gt_dir", type=str, default="../data/Lightwheel/all-uniform-100k", help="Path to gt pcd directory")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Evaluation results output directory")
    parser.add_argument("--meta_dir", type=str, default=None, help="Path to meta directory")
    parser.add_argument("--asset_type", type=str, default="obj", choices=["obj", "npy"], help="Asset type in mesh or points cloud")
    parser.add_argument("--result_dir", type=str, default="../data/release_converted_results", help="Path to converted result npz directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--resample_points", action="store_true", help="Resample points from the mesh")
    parser.add_argument("--num_points", type = int, default = 100000, help = "Number of points to sample from the mesh")
    parser.add_argument("--result_type", type=str, default="particulate", choices=["particulate", "custom"], help="Result type")
    args = parser.parse_args()

    if args.resample_points:
        assert args.asset_type == "obj", "Only obj asset type is supported for resampling points"
        assert args.meta_dir is not None, "Meta directory is required for resampling points"
        print(f"Resampling points from the mesh... Saving resampled points to {args.result_dir}")
        process_custom_prediction(meta_dir=args.meta_dir, num_points=args.num_points, save_dir=args.result_dir)

    main(
        result_dir=args.result_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        device=args.device,
        result_type = args.result_type,
    )