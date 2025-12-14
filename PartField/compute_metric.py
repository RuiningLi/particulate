import numpy as np
import json
from os.path import join
from typing import List
import os

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union != 0:
        return (intersection / union) * 100  
    else:
        return 0

def eval_single_gt_shape(gt_label, pred_masks):
    # gt: [N,], label index
    # pred: [B, N], B is the number of predicted parts, binary label
    unique_gt_label = np.unique(gt_label)
    best_ious = []
    for label in unique_gt_label:
        best_iou = 0
        if label == -1:
            continue
        for mask in pred_masks:
            iou = compute_iou(mask, gt_label == label)
            best_iou = max(best_iou, iou)
        best_ious.append(best_iou)
    return np.mean(best_ious)

def eval_whole_dataset(pred_folder, merge_parts=False):
    print(pred_folder)
    meta = json.load(open("/home/mikaelaangel/Desktop/data/PartObjaverse-Tiny_semantic.json", "r"))

    categories = meta.keys()
    results_per_cat = {}
    per_cat_mious = []
    overall_mious = []

    MAX_NUM_CLUSTERS = 20
    view_id = 0

    for cat in categories:
        results_per_cat[cat] = []
        for shape_id in meta[cat].keys():

            try:
                all_pred_labels = []
                for num_cluster in range(2, MAX_NUM_CLUSTERS):
                    ### load each label
                    fname_clustering = os.path.join(pred_folder, "cluster_out", str(shape_id) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2)) + ".npy"
                    pred_label = np.load(fname_clustering)
                    all_pred_labels.append(np.squeeze(pred_label))  

                all_pred_labels = np.array(all_pred_labels)

            except:
                continue

            pred_masks = []
            
            #### Path for PartObjaverseTiny Labels
            gt_labels_path = "PartObjaverse-Tiny_instance_gt"
            #################################

            gt_label = np.load(os.path.join(gt_labels_path, shape_id + ".npy"))

            if merge_parts:
                pred_masks = []
                for result in all_pred_labels:
                    pred = result
                    assert pred.shape[0] == gt_label.shape[0]
                    for label in np.unique(pred):
                        pred_masks.append(pred == label)
                miou = eval_single_gt_shape(gt_label, np.array(pred_masks))
                results_per_cat[cat].append(miou)
            else:
                best_miou = 0
                for result in all_pred_labels:
                    pred_masks = []
                    pred = result

                    for label in np.unique(pred):
                        pred_masks.append(pred == label)
                    miou = eval_single_gt_shape(gt_label, np.array(pred_masks))
                    best_miou = max(best_miou, miou)
                results_per_cat[cat].append(best_miou)
            
        print(np.mean(results_per_cat[cat]))
        per_cat_mious.append(np.mean(results_per_cat[cat]))
        overall_mious += results_per_cat[cat]
    print(np.mean(per_cat_mious))
    print(np.mean(overall_mious), len(overall_mious))

                
if __name__ == "__main__":
    eval_whole_dataset("dump_partobjtiny_clustering")

