import argparse
from ast import GtE
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
        xyz=results['points'],
        xyz_gt=gt['points'],
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

def process_prediction(
    obj_file: str,
    num_points: int,
    cache_dir: Optional[str] = None,
):
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, f"{obj_file.split('/')[-3]}.npz")

    meta_path = obj_file.replace("pred.obj", "pred.npz")
    meta = np.load(meta_path)
    face_part_id = meta["face_part_ids"]
    verts, faces = load_obj_raw_preserve(Path(obj_file))
    mesh = trimesh.Trimesh(verts, faces, process=False)
    # Sample points on the surface with normals
    points_uniform, face_indices = mesh.sample(num_points, return_index=True)
    part_ids = face_part_id[face_indices]
    results = {
        "points": points_uniform,
        "part_ids": part_ids,
        "motion_hierarchy": meta['motion_hierarchy'],
        "is_part_revolute": meta['is_part_revolute'],
        "is_part_prismatic": meta['is_part_prismatic'],
        "revolute_plucker": meta['revolute_plucker'],
        "revolute_range": meta['revolute_range'],
        "prismatic_axis": meta['prismatic_axis'],
        "prismatic_range": meta['prismatic_range'],
        "face_indices": face_indices,
    }
    if cache_dir is not None:
        np.savez(cache_file, **results)
    return results

def evaluate(
    gt_dir: str,
    result_dir: str,
    output_dir: str = "./inference_results",
    device: str = "cuda",
    num_points: int = 100000,
    cache_dir: Optional[str] = None,
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
    obj_files = glob.glob(os.path.join(result_dir, "**", "eval","*.obj"))
    for i, obj_file in enumerate(tqdm(obj_files, desc="Processing samples")):

        results = process_prediction(obj_file=obj_file, num_points=num_points, cache_dir=cache_dir)

        # check whether scaled 

        if not np.all(np.abs(results['points']) <= 0.5+1e-3):
            breakpoint()
        min_x, min_y, min_z = np.min(results['points'], axis=0)
        max_x, max_y, max_z = np.max(results['points'], axis=0)
        if not (np.abs(min_x + max_x)/2 < 1e-2 and np.abs(min_y + max_y)/2 < 1e-2 and np.abs(min_z + max_z)/2 < 1e-2):
            breakpoint()
        if not np.max(np.abs(results['points']) >= 0.5-1e-3):
            breakpoint()

        sample_name = obj_file.split("/")[-3]
        try:
            gt_file = os.path.join(gt_dir, f"{sample_name}.npz")
            gt = np.load(gt_file)

        except:
            # raise ValueError(f"GT file {gt_file} not found")
            print(f"GT file {gt_file} not found")
            continue

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
        'rest_per_part_avg_chamfer': np.round(np.mean([result['rest_per_part_avg_chamfer'] for result in eval_results]), 4),
        'rest_per_part_avg_giou': np.round(np.mean([result['rest_per_part_avg_giou'] for result in eval_results]), 4),
        'rest_per_part_avg_mIoU': np.round(np.mean([result['rest_per_part_avg_mIoU'] for result in eval_results]), 4),
    }
    overall_eval_results['fully_per_part_articulated_avg_chamfer'] = np.round(np.mean([result['fully_per_part_articulated_avg_chamfer'] for result in eval_results]), 4)
    overall_eval_results['fully_per_part_articulated_avg_giou'] = np.round(np.mean([result['fully_per_part_articulated_avg_giou'] for result in eval_results]), 4)
    overall_eval_results['fully_articulated_overall_chamfer_distances'] = np.round(np.mean([result['per_state_overall_chamfer_distances'][-1] for result in eval_results]), 4)
    json.dump(overall_eval_results, open(output_path.parent / f"{output_path.stem}_eval_overall.json", "w"), indent=4)

    overall_eval_results_nopunish = {
        'rest_per_part_avg_chamfer_nopunish': np.round(np.mean([result['rest_per_part_avg_chamfer_nopunish'] for result in eval_results]), 4),
        'rest_per_part_avg_giou_nopunish': np.round(np.mean([result['rest_per_part_avg_giou_nopunish'] for result in eval_results]), 4),
        'rest_per_part_avg_mIoU_nopunish': np.round(np.mean([result['rest_per_part_avg_mIoU_nopunish'] for result in eval_results]), 4),
    }
    
    overall_eval_results_nopunish['fully_per_part_articulated_avg_chamfer_nopunish'] = np.round(np.mean([result['per_state_chamfer_distances_nopunish'][-1] for result in eval_results]), 4)
    overall_eval_results_nopunish['fully_per_part_articulated_avg_giou_nopunish'] = np.round(np.mean([result['per_state_giou_nopunish'][-1] for result in eval_results]), 4)
    overall_eval_results_nopunish['fully_articulated_overall_chamfer_distances_nopunish'] = np.round(np.mean([result['per_state_overall_chamfer_distances'][-1] for result in eval_results]), 4)
    json.dump(overall_eval_results_nopunish, open(output_path.parent / f"{output_path.stem}_eval_overall_nopunish.json", "w"), indent=4)
    print(f"Inference completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on Articulate3D model")
    parser.add_argument("--gt_dir", type=str, default="dataset/Lightwheel_uniform-100k", help="Path to gt pcd directory")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to result directory")
    parser.add_argument("--output_dir", type=str, default="eval_result", help="Evaluation results output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_points", type = int, default =100000, help = "Number of points to sample from the mesh")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache directory, if not provided, will process all obj files but not cache them")
    args = parser.parse_args()

    evaluate(
        result_dir=args.result_dir,
        gt_dir=args.gt_dir,
        num_points=args.num_points,
        output_dir=args.output_dir,
        device=args.device,
        cache_dir=args.cache_dir,
    )
