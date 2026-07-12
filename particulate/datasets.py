import os
import math
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

from particulate.articulation_utils import (
    plucker_to_axis_point,
    axis_point_to_plucker,
    articulate_points,
)
from particulate.data_utils import normalize_meshes
from particulate.articulation_utils import shift_axes_plucker


def get_gt_motion_params(
    link_axes_plucker: np.ndarray,
    link_range: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def collate_fn(batch: List[Dict[str, Any]], max_parts: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that dynamically pads part-related tensors to the maximum number of parts in the batch.

    Args:
        batch: List of dataset samples, each containing tensors with potentially different part dimensions

    Returns:
        Dictionary with batched tensors, where part-related tensors are padded to max_parts
    """
    if max_parts is None:
        # Find the maximum number of parts in this batch
        max_parts = max(sample['num_valid_parts'].item() for sample in batch)

    # Initialize the batched dictionary
    batched = {}

    # Handle tensors that don't need padding (same shape across all samples)
    same_shape_keys = ['xyz', 'normals', 'feats', 'part_ids', 'all_points', 'all_part_ids', 'gt_closest_point_on_axis']
    for key in same_shape_keys:
        if key in batch[0]:
            batched[key] = torch.stack([sample[key] for sample in batch])

    # Handle part-related tensors that need padding
    part_related_keys = [
        'gt_part_motion_class',
        'gt_revolute_plucker',
        'gt_prismatic_axis',
        'gt_revolute_range',
        'gt_prismatic_range',
        'query_xyz',
        'query_feats'
    ]

    for key in part_related_keys:
        if key in batch[0]:
            # Get the shape of this tensor (excluding the first dimension which is num_parts)
            sample_shape = batch[0][key].shape[1:] if len(batch[0][key].shape) > 1 else ()

            # Create a list to hold padded tensors
            padded_tensors = []

            for sample in batch:
                num_parts = sample['num_valid_parts'].item()
                tensor = sample[key]

                if len(tensor.shape) == 1:
                    # 1D tensor (e.g., gt_part_motion_class)
                    padded = torch.zeros(max_parts, dtype=tensor.dtype, device=tensor.device)
                    padded[:num_parts] = tensor
                elif len(tensor.shape) == 2:
                    # 2D tensor (e.g., gt_revolute_plucker)
                    padded = torch.zeros(max_parts, tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
                    padded[:num_parts, :] = tensor
                else:
                    # Higher dimensional tensor
                    padded = torch.zeros((max_parts,) + sample_shape, dtype=tensor.dtype, device=tensor.device)
                    padded[:num_parts] = tensor

                padded_tensors.append(padded)

            batched[key] = torch.stack(padded_tensors)

    # Handle part_structure_matrix separately since it's a square matrix
    if 'part_structure_matrix' in batch[0]:
        padded_tensors = []
        for sample in batch:
            num_parts = sample['num_valid_parts'].item()
            tensor = sample['part_structure_matrix']

            # Create a square matrix of size max_parts x max_parts
            padded = torch.zeros(max_parts, max_parts, dtype=tensor.dtype, device=tensor.device)
            padded[:num_parts, :num_parts] = tensor

            padded_tensors.append(padded)

        batched['part_structure_matrix'] = torch.stack(padded_tensors)

    # Handle num_valid_parts (just stack them)
    batched['num_valid_parts'] = torch.stack([sample['num_valid_parts'] for sample in batch])

    # Handle text_labels (list of strings)
    if 'text_labels' in batch[0]:
        batched['text_labels'] = [sample['text_labels'] for sample in batch]

    return batched


def sample_points(sharp_point_flag: np.ndarray, num_points: int, sharp_point_ratio: float) -> np.ndarray:
    """
    Sample point indices with stratified sampling based on sharp_point_ratio.

    Args:
        sharp_point_flag: Boolean array indicating which points are sharp
        num_points: Total number of points to sample
        sharp_point_ratio: Desired ratio of sharp points (between 0 and 1)

    Returns:
        Array of sampled indices, shuffled to avoid ordering bias
    """
    sharp_indices = np.where(sharp_point_flag)[0]
    non_sharp_indices = np.where(~sharp_point_flag)[0]

    num_sharp_desired = int(num_points * sharp_point_ratio)
    num_non_sharp_desired = num_points - num_sharp_desired

    # Handle edge cases
    num_sharp_available = len(sharp_indices)
    num_non_sharp_available = len(non_sharp_indices)

    if num_sharp_available >= num_sharp_desired and num_non_sharp_available >= num_non_sharp_desired:
        # Normal case: enough points in both sets
        sampled_sharp = np.random.choice(sharp_indices, num_sharp_desired, replace=False)
        sampled_non_sharp = np.random.choice(non_sharp_indices, num_non_sharp_desired, replace=False)
    elif num_sharp_available < num_sharp_desired:
        # Not enough sharp points: sample all sharp points and fill rest from non-sharp
        sampled_sharp = sharp_indices
        remaining_needed = num_points - num_sharp_available
        sampled_non_sharp = np.random.choice(non_sharp_indices, remaining_needed, replace=False)
    else:
        # Not enough non-sharp points: sample all non-sharp points and fill rest from sharp
        sampled_non_sharp = non_sharp_indices
        remaining_needed = num_points - num_non_sharp_available
        sampled_sharp = np.random.choice(sharp_indices, remaining_needed, replace=False)

    indices = np.concatenate([sampled_sharp, sampled_non_sharp])
    np.random.shuffle(indices)  # Shuffle to avoid ordering bias

    return indices


class WeightedConcatSampler(torch.utils.data.Sampler):
    """
    Sampler that samples from a concatenated dataset with specified sampling rates for each sub-dataset.
    Supports distributed training by partitioning samples across ranks.
    """
    def __init__(
        self,
        dataset_sizes: List[int],
        sampling_rates: List[float],
        num_samples: int,
        generator=None,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset_sizes = dataset_sizes
        self.sampling_rates = sampling_rates
        self.num_samples = num_samples
        self.generator = generator
        self.cumulative_sizes = [0] + list(np.cumsum(dataset_sizes))

        # Distributed training support
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Calculate samples per replica
        self.total_size = num_samples
        self.num_samples_per_replica = int(math.ceil(self.total_size / self.num_replicas))

    def __len__(self):
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler. This ensures different shuffling for each epoch
        when using distributed training.
        """
        self.epoch = epoch

    def __iter__(self):
        # Create generator with epoch-specific seed for distributed training
        if self.shuffle:
            # Each replica gets a different seed based on rank and epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank * 10000)
        else:
            g = self.generator if self.generator is not None else torch.Generator()

        # Normalize sampling rates
        total_rate = sum(self.sampling_rates)
        normalized_rates = [r / total_rate for r in self.sampling_rates]

        # Sample dataset indices according to sampling rates
        # Generate enough samples for all replicas
        total_samples_needed = self.num_samples_per_replica * self.num_replicas
        dataset_indices = torch.multinomial(
            torch.tensor(normalized_rates, dtype=torch.float32),
            num_samples=total_samples_needed,
            replacement=True,
            generator=g
        )

        # For each dataset index, sample a random example from that dataset
        indices = []
        for dataset_idx in dataset_indices:
            dataset_idx = dataset_idx.item()
            dataset_start = self.cumulative_sizes[dataset_idx]
            dataset_size = self.dataset_sizes[dataset_idx]

            # Sample a random index from this dataset
            random_offset = torch.randint(0, dataset_size, (1,), generator=g).item()
            indices.append(dataset_start + random_offset)

        # Partition indices for this replica
        indices = indices[self.rank:total_samples_needed:self.num_replicas]

        return iter(indices)


class OntheFlyArticulated3DDataset(Dataset):
    """
    Pre-processed dataset of articulated 3D objects.
    """
    def __init__(
        self,
        root: str,
        num_points: int,
        feat_root: Optional[str] = None,
        num_poses: int = 12,
        include_point_prompts: bool = True,
        include_text_labels: bool = False,
        sharp_point_ratio: float = 0.5,
        resting_state_prob: float = 0.1,
        max_parts: int = 128,
        split_str: str = '',
        cached_files: Optional[List[str]] = None,
        use_augmentation: bool = False,
        normalize_points: bool = False,
        compute_feat_on_the_fly: bool = False,
        normal_dropout_rate: float = 0.0,
        random_scale = [1, 1],
        random_translation_std = 0,
        random_rotation_deg_std = 0
    ):
        self.root = root
        self.feat_root = feat_root
        if cached_files is not None:
            self.data = sorted([f for f in sorted(os.listdir(root)) if f in cached_files])
        else:
            self.data = sorted([f for f in sorted(os.listdir(root)) if (f.endswith('.npz') and split_str in f)])
        self.num_points = num_points
        self.include_point_prompts = include_point_prompts
        self.include_text_labels = include_text_labels

        self.num_poses = num_poses
        self.max_parts = max_parts
        self.use_augmentation = use_augmentation
        self.sharp_point_ratio = sharp_point_ratio
        self.resting_state_prob = resting_state_prob
        self.normalize_points = normalize_points
        self.compute_feat_on_the_fly = compute_feat_on_the_fly
        self.normal_dropout_rate = normal_dropout_rate
        self.random_scale = random_scale
        self.random_translation_std = random_translation_std
        self.random_rotation_deg_std = random_rotation_deg_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_retry = 5
        last_error = None
        for _ in range(max_retry):
            try:
                return self._getitem(idx)
            except Exception as e:
                last_error = e
                print(f"Error loading {self.data[idx]} at index {idx}: {e}")
                idx = np.random.randint(0, len(self.data))
        raise RuntimeError(f"Failed to load a sample after {max_retry} attempts") from last_error

    def _augment_points(
        self,
        points_or_axes: np.ndarray,
        rotation_z: float
    ) -> np.ndarray:
        assert points_or_axes.shape[-1] == 3
        rotation_matrix_z = np.array([
            [np.cos(rotation_z), -np.sin(rotation_z), 0],
            [np.sin(rotation_z), np.cos(rotation_z), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        points_or_axes = points_or_axes @ rotation_matrix_z.T
        return points_or_axes

    def _getitem(self, idx):
        data_path = os.path.join(self.root, self.data[idx])

        if self.use_augmentation:
            rotation_z = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])

        with np.load(data_path, allow_pickle=True) as data:
            num_parts = data['link_axes_plucker'].shape[0]
            if num_parts > self.max_parts:
                raise RuntimeError(f"num_parts {num_parts} is greater than max_parts {self.max_parts}")

            all_points = data['points']
            all_normals = data['normals']
            all_part_ids = data['point_to_bone']
            if self.use_augmentation:
                all_points = self._augment_points(all_points, rotation_z)
                all_normals = self._augment_points(all_normals, rotation_z)

            sharp_point_flag = data['point_from_sharp']
            num_total_points = all_points.shape[0]

            if isinstance(self.num_points, list):
                num_points = np.random.randint(self.num_points[0], self.num_points[1])
            else:
                num_points = self.num_points
            assert num_points <= num_total_points, \
                f"num_points {num_points} is greater than number of cached points {num_total_points}"

            # Sample indices with stratified sampling based on sharp_point_ratio
            indices = sample_points(sharp_point_flag, num_points, self.sharp_point_ratio)

            points = all_points[indices]
            normals = all_normals[indices]
            part_ids = all_part_ids[indices]

            if not self.compute_feat_on_the_fly:
                pose_idx = np.random.randint(0, self.num_poses)
                feat_path = os.path.join(self.feat_root, self.data[idx].replace(".npz", f"_pos{pose_idx:02d}.npy"))
                feats = np.load(feat_path)[indices]

            link_axes_plucker = data['link_axes_plucker']
            link_range = data['link_range']
            bone_structure = data['bone_structure']
            if self.include_text_labels:
                text_labels = data['text_labels']
            else:
                text_labels = None

            if self.include_point_prompts:
                query_xyz = np.zeros((num_parts, 3), dtype=np.float32)  # One query (3D location) per part
                query_feats = np.zeros((num_parts, feats.shape[-1]), dtype=np.float32)  # One query (3D location) per part

                # Sample one point per part as a prompt
                for pid in range(num_parts):
                    # Find points that belong to this part
                    part_mask = data['point_to_bone'] == pid
                    if part_mask.any():
                        # Sample one point from this part
                        part_point_indices = np.where(part_mask)[0]
                        sampled_idx = np.random.choice(part_point_indices)
                        query_xyz[pid] = all_points[sampled_idx]
                        query_feats[pid] = data['point_features'][sampled_idx]
                    else:
                        # If no points belong to this part, use zero vector
                        query_xyz[pid] = np.zeros(3, dtype=np.float32)
                        query_feats[pid] = np.zeros(feats.shape[-1], dtype=np.float32)

        assert num_parts == link_axes_plucker.shape[0] == link_range.shape[0], \
            f"num_parts {num_parts} does not match link_axes_plucker.shape {link_axes_plucker.shape}, link_range.shape {link_range.shape}"

        (
            gt_part_motion_class,
            gt_revolute_plucker, gt_prismatic_axis,
            gt_revolute_range, gt_prismatic_range
        ) = get_gt_motion_params(link_axes_plucker, link_range)

        if self.use_augmentation:
            revolute_flag = np.linalg.norm(gt_revolute_plucker[..., :3], axis=-1) > 1e-8
            prismatic_flag = np.linalg.norm(gt_prismatic_axis, axis=-1) > 1e-8
            gt_revolute_axis, gt_revolute_point = plucker_to_axis_point(gt_revolute_plucker[revolute_flag])
            gt_revolute_axis = self._augment_points(gt_revolute_axis, rotation_z)
            gt_revolute_point = self._augment_points(gt_revolute_point, rotation_z)
            gt_revolute_plucker[revolute_flag] = axis_point_to_plucker(gt_revolute_axis, gt_revolute_point)
            gt_prismatic_axis[prismatic_flag] = self._augment_points(gt_prismatic_axis[prismatic_flag], rotation_z)

        # On the fly sample pose and articulate points
        # 1) Sample a state for each part
        if np.random.uniform(0, 1) < self.resting_state_prob:
            articulation_state = np.zeros(num_parts, dtype=np.float32)
        else:
            articulation_state = np.array([
                0 if np.random.uniform(0, 1) < self.resting_state_prob else np.random.uniform(0, 1)
                for _ in range(num_parts)
            ])
        # 2) Articulate points
        points, gt_revolute_plucker, gt_prismatic_axis, gt_revolute_range, gt_prismatic_range, normals = articulate_points(
            xyz=points if not self.compute_feat_on_the_fly else np.concatenate([points, all_points], axis=0),
            part_ids=part_ids if not self.compute_feat_on_the_fly else np.concatenate([part_ids, all_part_ids], axis=0),
            motion_hierarchy=bone_structure,
            is_part_revolute=np.logical_or(gt_part_motion_class == 1, gt_part_motion_class == 3),
            is_part_prismatic=np.logical_or(gt_part_motion_class == 2, gt_part_motion_class == 3),
            revolute_plucker=gt_revolute_plucker,
            revolute_range=gt_revolute_range,
            prismatic_axis=gt_prismatic_axis,
            prismatic_range=gt_prismatic_range,
            articulation_state=articulation_state,
            normals=normals if not self.compute_feat_on_the_fly else np.concatenate([normals, all_normals], axis=0),
        )
        # 3) Normalize points to [-0.5, 0.5]
        if self.normalize_points:
            points, x_center, y_center, z_center, scale = normalize_meshes([points])
            points = points[0]
            gt_prismatic_range = gt_prismatic_range * scale
            revolute_flag = np.linalg.norm(gt_revolute_plucker[..., :3], axis=-1) > 1e-8
            gt_revolute_plucker[revolute_flag] = shift_axes_plucker(gt_revolute_plucker[revolute_flag], x_center, y_center, z_center, scale)

        # 4) Random scale, translation, and rotation
        scale = np.random.uniform(self.random_scale[0], self.random_scale[1])
        rotation_deg = np.random.randn(3) * self.random_rotation_deg_std
        translation = np.random.randn(3) * self.random_translation_std
        # 4.1) Scale
        points = points * scale
        gt_prismatic_range = gt_prismatic_range * scale
        gt_revolute_plucker[..., 3:] = gt_revolute_plucker[..., 3:] * scale
        # 4.2) Rotation
        rot_matrix = Rotation.from_euler('ZYX', rotation_deg, degrees=True).as_matrix()
        points = points @ rot_matrix.T
        normals = normals @ rot_matrix.T
        gt_prismatic_axis = gt_prismatic_axis @ rot_matrix.T
        gt_revolute_axis, gt_revolute_point = plucker_to_axis_point(gt_revolute_plucker)
        gt_revolute_axis = gt_revolute_axis @ rot_matrix.T
        gt_revolute_point = gt_revolute_point @ rot_matrix.T
        gt_revolute_plucker = axis_point_to_plucker(gt_revolute_axis, gt_revolute_point)
        # 4.3) Translation
        points = points + translation
        gt_revolute_axis, gt_revolute_point = plucker_to_axis_point(gt_revolute_plucker)
        gt_revolute_point = gt_revolute_point + translation
        gt_revolute_plucker = axis_point_to_plucker(gt_revolute_axis, gt_revolute_point)

        if self.compute_feat_on_the_fly:
            points, all_points = points[:num_points], points[num_points:]
            normals, all_normals = normals[:num_points], normals[num_points:]

        # Convert bone_structure to a adjacency matrix
        part_structure_matrix = np.zeros((num_parts, num_parts), dtype=np.bool_)
        for parent_id, child_id in bone_structure:
            if 0 <= parent_id and parent_id < num_parts and 0 <= child_id and child_id < num_parts:
                part_structure_matrix[parent_id, child_id] = True

        gt_closest_point_on_axis = np.zeros((num_points, 3), dtype=np.float32)
        for pid in range(num_parts):
            if np.logical_or(gt_part_motion_class[pid] == 1, gt_part_motion_class[pid] == 3):
                plucker = gt_revolute_plucker[pid]
                part_points = points[part_ids == pid]
                axis_dir, axis_point = plucker_to_axis_point(plucker)
                closest_point_on_axis = axis_point + axis_dir * np.dot(part_points - axis_point, axis_dir)[:, None]
                gt_closest_point_on_axis[part_ids == pid] = closest_point_on_axis

        if np.random.uniform(0, 1) < self.normal_dropout_rate:
            normals = np.zeros_like(normals)

        ret = dict(
            xyz=torch.from_numpy(points).float(),
            normals=torch.from_numpy(normals).float(),
            part_ids=torch.from_numpy(part_ids).long(),
            num_valid_parts=torch.tensor(num_parts, dtype=torch.long),
            gt_part_motion_class=torch.from_numpy(gt_part_motion_class).long(),
            gt_revolute_plucker=torch.from_numpy(gt_revolute_plucker).float(),
            gt_prismatic_axis=torch.from_numpy(gt_prismatic_axis).float(),
            gt_revolute_range=torch.from_numpy(gt_revolute_range).float(),
            gt_prismatic_range=torch.from_numpy(gt_prismatic_range).float(),
            gt_closest_point_on_axis=torch.from_numpy(gt_closest_point_on_axis).float(),
            part_structure_matrix=torch.from_numpy(part_structure_matrix),
            text_labels=text_labels
        )
        if not self.compute_feat_on_the_fly:
            ret['feats'] = torch.from_numpy(feats).float()
        else:
            ret['all_points'] = torch.from_numpy(all_points).float()
            ret['all_part_ids'] = torch.from_numpy(all_part_ids).long()
        if self.include_point_prompts:
            ret['query_xyz'] = torch.from_numpy(query_xyz).float()
            ret['query_feats'] = torch.from_numpy(query_feats).float()

        return ret

    @staticmethod
    def get_collate_fn(max_parts: Optional[int] = None):
        """Return the custom collate function for this dataset."""
        return lambda batch: collate_fn(batch, max_parts)
