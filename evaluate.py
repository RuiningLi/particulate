import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from particulate.data_utils import load_obj_raw_preserve
from particulate.evaluation_utils import evaluate_articulate_result

@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.bfloat16)
def evaluate_inference_results(
    results: Dict[str, Any],
    gt: Dict[str, Any],
    output_json_path: Path,
    hungarian_matching_cost_type = "cdist",  # "cdist"or "chamfer"
):
    """Evaluate inference results."""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    eval_results = evaluate_articulate_result(
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
    json.dump(eval_results, open(output_json_path, "w"), indent=4)
    return eval_results


def process_prediction(pred_dir: Path, num_points: int):
    """Augment the prediction results with uniformly sampled points on the surface."""
    results = dict(np.load(pred_dir / "pred.npz"))
    face_part_id = results["face_part_ids"]
    verts, faces = load_obj_raw_preserve(pred_dir / "pred.obj")
    mesh = trimesh.Trimesh(verts, faces, process=False)
    points_uniform, face_indices = mesh.sample(num_points, return_index=True)
    part_ids = face_part_id[face_indices]
    return {**results, "points": points_uniform, "part_ids": part_ids}


def assert_points_normalized(points: np.ndarray) -> None:
    """Ensure points are centered at origin (bbox) and within [-0.5, 0.5]."""
    assert np.allclose(points.min(0) + points.max(0), 0, atol=1e-4), "Bounding box not centered at origin"
    assert 0.5 - 1e-4 <= np.abs(points).max() <= 0.5 + 1e-4, f"Bounding box not as expected [-0.5, 0.5]: {np.abs(points).max()}"


def evaluate(
    gt_dir: Path,
    result_dir: Path,
    output_dir: Path,
    num_points: int = 100000
):
    """Evaluate all inference results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_results = []
    
    for pred_dir in tqdm(result_dir.glob("*/eval"), desc="Processing samples"):
        results = process_prediction(pred_dir=pred_dir, num_points=num_points)
        assert_points_normalized(results['points'])

        sample_name = pred_dir.parent.name
        try:
            gt = dict(np.load(gt_dir / f"{sample_name}.npz"))
        except FileNotFoundError as e:
            print(f"Corresponding ground-truth file of {sample_name} not found: {e}")
            continue

        eval_json = (output_dir / sample_name).with_suffix(".json")
        if eval_json.exists():
            with open(eval_json, "r") as f:
                eval_result = json.load(f)
        else:
            eval_result = evaluate_inference_results(results, gt, eval_json, hungarian_matching_cost_type='cdist')
        eval_results.append(eval_result)

    overall_eval_results = {
        'rest_per_part_avg_chamfer': np.round(np.mean([result['rest_per_part_avg_chamfer'] for result in eval_results]), 4),
        'rest_per_part_avg_giou': np.round(np.mean([result['rest_per_part_avg_giou'] for result in eval_results]), 4),
        'rest_per_part_avg_mIoU': np.round(np.mean([result['rest_per_part_avg_mIoU'] for result in eval_results]), 4),
    }
    overall_eval_results['fully_per_part_articulated_avg_chamfer'] = np.round(np.mean([result['fully_per_part_articulated_avg_chamfer'] for result in eval_results]), 4)
    overall_eval_results['fully_per_part_articulated_avg_giou'] = np.round(np.mean([result['fully_per_part_articulated_avg_giou'] for result in eval_results]), 4)
    overall_eval_results['fully_articulated_overall_chamfer_distances'] = np.round(np.mean([result['per_state_overall_chamfer_distances'][-1] for result in eval_results]), 4)
    json.dump(overall_eval_results, open(output_dir / "OVERALL_EVAL_RESULTS.json", "w"), indent=4)

    overall_eval_results_nopunish = {
        'rest_per_part_avg_chamfer_nopunish': np.round(np.mean([result['rest_per_part_avg_chamfer_nopunish'] for result in eval_results]), 4),
        'rest_per_part_avg_giou_nopunish': np.round(np.mean([result['rest_per_part_avg_giou_nopunish'] for result in eval_results]), 4),
        'rest_per_part_avg_mIoU_nopunish': np.round(np.mean([result['rest_per_part_avg_mIoU_nopunish'] for result in eval_results]), 4),
    }
    
    overall_eval_results_nopunish['fully_per_part_articulated_avg_chamfer_nopunish'] = np.round(np.mean([result['per_state_chamfer_distances_nopunish'][-1] for result in eval_results]), 4)
    overall_eval_results_nopunish['fully_per_part_articulated_avg_giou_nopunish'] = np.round(np.mean([result['per_state_giou_nopunish'][-1] for result in eval_results]), 4)
    overall_eval_results_nopunish['fully_articulated_overall_chamfer_distances_nopunish'] = np.round(np.mean([result['per_state_overall_chamfer_distances'][-1] for result in eval_results]), 4)
    json.dump(overall_eval_results_nopunish, open(output_dir / "OVERALL_EVAL_RESULTS_NOPUNISH.json", "w"), indent=4)
    print(f"Inference completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on Articulate3D model")
    parser.add_argument("--gt_dir", type=Path, required=True, help="Directory containing all cached ground-truth files (an npz file for each asset)")
    parser.add_argument("--result_dir", type=Path, required=True, help="Directory containing all predictions (an npz file and an obj file for each asset)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save evaluation results")
    parser.add_argument("--num_points", type=int, default=100_000, help = "Number of points to sample for Chamfer Distance computation")
    args = parser.parse_args()

    evaluate(gt_dir=args.gt_dir, result_dir=args.result_dir, output_dir=args.output_dir, num_points=args.num_points)
