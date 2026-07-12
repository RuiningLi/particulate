"""Train Particulate from cached articulated point clouds."""

import argparse
import datetime
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from particulate import datasets, models


logger = get_logger(__name__)


def create_dataset(config: Dict[str, Any], split: str) -> datasets.OntheFlyArticulated3DDataset:
    """Build one dataset while applying its optional train/test split file."""
    cached_files = None
    split_file = config.get("data_split_file")
    if split_file:
        split_data = OmegaConf.load(split_file)
        cached_files = list(split_data[split])

    return datasets.OntheFlyArticulated3DDataset(
        root=config["root"],
        feat_root=config.get("feat_root"),
        num_points=config["num_points"],
        num_poses=config.get("num_poses", 12),
        include_point_prompts=False,
        include_text_labels=False,
        sharp_point_ratio=config.get("sharp_point_ratio", 0.5),
        resting_state_prob=config.get("resting_state_prob", 0.25),
        max_parts=config.get("max_parts", 16),
        cached_files=cached_files,
        use_augmentation=config.get("use_augmentation", False) and split == "train",
        normalize_points=config.get("normalize_points", True),
        compute_feat_on_the_fly=config.get("compute_feat_on_the_fly", True),
        normal_dropout_rate=config.get("normal_dropout_rate", 0.0),
        random_scale=config.get("random_scale", [1.0, 1.0]),
        random_translation_std=config.get("random_translation_std", 0.0),
        random_rotation_deg_std=config.get("random_rotation_deg_std", 0.0),
    )


def build_model(config: DictConfig) -> torch.nn.Module:
    constructors = {"S": models.PAT_S, "B": models.PAT_B, "L": models.PAT_L, "XL": models.PAT_XL}
    try:
        constructor = constructors[config.model_type]
    except KeyError as error:
        raise ValueError(f"Unknown model type: {config.model_type}") from error
    return constructor(
        input_dim=config.model_input_dim,
        dropout=config.model_dropout,
        use_normals=config.model_use_normals,
        max_parts=config.model_max_parts,
        use_part_id_embedding=config.model_use_part_id_embedding,
        use_raw_coords=config.model_use_raw_coords,
        use_point_features_for_motion_decoding=config.model_use_point_features_for_motion_decoding,
        num_mask_hypotheses=config.model_num_mask_hypotheses,
        motion_representation=config.model_motion_representation,
    )


def weighted_loss(losses: Dict[str, torch.Tensor], config: DictConfig) -> torch.Tensor:
    weights = {
        "point_mask_loss": config.loss_weight_point_mask,
        "dice_loss": config.loss_weight_dice,
        "motion_hierarchy_loss": config.loss_weight_motion_hierarchy,
        "part_motion_classification_loss": config.loss_weight_part_motion_classification,
        "part_motion_axis_loss_revolute": config.loss_weight_part_motion_axis_revolute,
        "part_motion_axis_loss_prismatic": config.loss_weight_part_motion_axis_prismatic,
        "part_motion_range_loss_revolute": config.loss_weight_part_motion_range_revolute,
        "part_motion_range_loss_prismatic": config.loss_weight_part_motion_range_prismatic,
        "point_closest_point_on_axis_loss": config.loss_weight_point_closest_point_on_axis,
    }
    return sum(weights[name] * value for name, value in losses.items())


def save_checkpoint(accelerator: Accelerator, model: torch.nn.Module, run_dir: Path, step: Optional[int]) -> None:
    name = "final_model" if step is None else f"checkpoint-{step:07d}"
    output = run_dir / name
    accelerator.save_state(output)
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), output / "model.pt")


def train(config: DictConfig) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=None if config.logger_type == "none" else config.logger_type,
        project_dir=config.output_dir,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(config.seed, device_specific=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if accelerator.is_main_process else None
    timestamp = broadcast_object_list([timestamp], from_process=0)[0]
    run_dir = Path(config.output_dir) / f"{config.exp_name}-{config.model_type}_{timestamp}"
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(config, run_dir / "config.yaml")
    accelerator.wait_for_everyone()

    shared = OmegaConf.to_container(config.dataset_args, resolve=True)
    dataset_configs = []
    for item in shared.pop("datasets"):
        dataset_configs.append({**shared, **item})
    train_sets = [create_dataset(item, "train") for item in dataset_configs]
    if not train_sets or any(len(dataset) == 0 for dataset in train_sets):
        raise ValueError("Each configured training dataset must contain at least one cached .npz file")
    train_dataset = ConcatDataset(train_sets)
    sampler = datasets.WeightedConcatSampler(
        [len(dataset) for dataset in train_sets],
        [item.get("sampling_rate", 1.0) for item in dataset_configs],
        num_samples=config.samples_per_epoch,
        seed=config.seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=train_sets[0].get_collate_fn(config.model_max_parts),
    )

    model = build_model(config)
    learning_rate = config.learning_rate
    if config.scale_lr:
        learning_rate *= accelerator.num_processes * config.gradient_accumulation_steps * config.batch_size
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay, eps=config.adam_epsilon,
    )
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, config.lr_warmup_steps, config.max_train_steps, min_lr_rate=0.1,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    if config.dataset_args.compute_feat_on_the_fly:
        from partfield_utils import get_partfield_model, obtain_partfield_feats

        partfield_model = get_partfield_model(str(accelerator.device))
    else:
        partfield_model = None

    if config.logger_type != "none":
        accelerator.init_trackers("particulate", config=OmegaConf.to_container(config, resolve=True), init_kwargs={"wandb": {"name": config.exp_name}})

    step = 0
    if config.resume_from_checkpoint:
        accelerator.load_state(config.resume_from_checkpoint)
        step = int(Path(config.resume_from_checkpoint).name.rsplit("-", 1)[-1])
    progress = tqdm(total=config.max_train_steps, initial=step, disable=not accelerator.is_local_main_process)
    epoch = 0
    while step < config.max_train_steps:
        sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            if partfield_model is not None:
                with torch.no_grad():
                    feats = obtain_partfield_feats(partfield_model, batch["all_points"], batch["xyz"])
            else:
                feats = batch["feats"]
            with accelerator.accumulate(model):
                losses, _ = model(
                    xyz=batch["xyz"], feats=feats, normals=batch.get("normals"),
                    part_ids=batch["part_ids"], num_valid_parts=batch["num_valid_parts"],
                    part_structure_matrix=batch["part_structure_matrix"],
                    gt_part_motion_class=batch["gt_part_motion_class"],
                    gt_revolute_plucker=batch["gt_revolute_plucker"],
                    gt_revolute_range=batch["gt_revolute_range"],
                    gt_prismatic_axis=batch["gt_prismatic_axis"],
                    gt_prismatic_range=batch["gt_prismatic_range"],
                    gt_closest_point_on_axis=batch["gt_closest_point_on_axis"],
                    run_matching=True,
                )
                loss = weighted_loss(losses, config)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                progress.update(1)
                if step % config.log_interval == 0:
                    metrics = {name: accelerator.gather(value.detach()).mean().item() for name, value in losses.items()}
                    metrics.update(loss=accelerator.gather(loss.detach()).mean().item(), lr=scheduler.get_last_lr()[0])
                    accelerator.log(metrics, step=step)
                    progress.set_postfix(loss=f"{metrics['loss']:.4f}")
                if step % config.ckpt_interval == 0:
                    save_checkpoint(accelerator, model, run_dir, step)
                if step >= config.max_train_steps:
                    break
    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, model, run_dir, None)
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-particulate-B.yaml")
    args = parser.parse_args()
    train(OmegaConf.load(args.config))


if __name__ == "__main__":
    main()
