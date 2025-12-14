from partfield.config import default_argument_parser, setup
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import glob
import os
import numpy as np
import random
import zipfile

from partfield.model.PVCNN.encoder_pc import sample_triplane_feat

def predict(cfg):
    seed_everything(cfg.seed)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    checkpoint_callbacks = [ModelCheckpoint(
        monitor="train/current_epoch",
        dirpath=cfg.output_dir,
        filename="{epoch:02d}",
        save_top_k=100,
        save_last=True,
        every_n_epochs=cfg.save_every_epoch,
        mode="max",
        verbose=True
    )]

    trainer = Trainer(devices=-1,
                      accelerator="gpu",
                      precision="16-mixed",
                      strategy=DDPStrategy(find_unused_parameters=True),
                      max_epochs=cfg.training_epochs,
                      log_every_n_steps=1,
                      limit_train_batches=3500,
                      limit_val_batches=None,
                      callbacks=checkpoint_callbacks
                     )

    from partfield.model_trainer_pvcnn_only_demo import Model
    model = Model(cfg)        

    if cfg.remesh_demo:
        cfg.n_point_per_face = 10

    trainer.predict(model, ckpt_path=cfg.continue_ckpt)
        
def main():
    from tqdm import tqdm

    parser = default_argument_parser()
    parser.add_argument('--num_jobs', type=int, default=1, help='Total number of parallel jobs')
    parser.add_argument('--job_id', type=int, default=0, help='Current job ID (0 to num_jobs-1)')
    args = parser.parse_args()
    cfg = setup(args, freeze=False)
    cfg.is_pc = True

    # Validate job arguments
    if args.job_id >= args.num_jobs:
        raise ValueError(f"job_id ({args.job_id}) must be less than num_jobs ({args.num_jobs})")
    if args.job_id < 0:
        raise ValueError(f"job_id ({args.job_id}) must be >= 0")

    from partfield.model_trainer_pvcnn_only_demo import Model
    model = Model.load_from_checkpoint(cfg.continue_ckpt, cfg=cfg)
    model.eval()
    model.to('cuda')

    encode_pc_root = "/scratch/shared/beegfs/ruining/data/articulate-3d/Lightwheel/all-uniform-100k-singlestate-pts"
    decode_pc_root = "/scratch/shared/beegfs/ruining/data/articulate-3d/Lightwheel/all-sharp50pct-40k-singlestate-pts"
    dest_feat_root = "/scratch/shared/beegfs/ruining/data/articulate-3d/Lightwheel/all-sharp50pct-40k-singlestate-feats"

    # Create destination directory
    os.makedirs(dest_feat_root, exist_ok=True)
    
    encode_files = sorted(glob.glob(os.path.join(encode_pc_root, "*.npy")))
    decode_files = sorted(glob.glob(os.path.join(decode_pc_root, "*.npy")))
    
    # Filter files for this job
    job_files = [pair for i, pair in enumerate(zip(encode_files, decode_files)) if i % args.num_jobs == args.job_id]
    
    print(f"Job {args.job_id}/{args.num_jobs}: Processing {len(job_files)}/{len(encode_files)} files")

    num_bad_zip, num_failed_others = 0, 0
    for encode_file, decode_file in tqdm(job_files, desc=f"Job {args.job_id}"):
        try:
            # Get UID from decode file (the one we're extracting features for)
            uid = os.path.basename(decode_file).split('.')[0]
            assert uid == os.path.basename(encode_file).split('.')[0]

            dest_feat_file = os.path.join(dest_feat_root, f"{uid}.npy")
            if os.path.exists(dest_feat_file):
                continue

            # Load both encode and decode point clouds
            encode_pc = np.load(encode_file)
            decode_pc = np.load(decode_file)
            
            # Validate input data
            if np.isnan(encode_pc).any() or np.isnan(decode_pc).any():
                print(f"Skipping {uid}: NaN values in point cloud")
                num_failed_others += 1
                continue
            if np.isinf(encode_pc).any() or np.isinf(decode_pc).any():
                print(f"Skipping {uid}: Inf values in point cloud")
                num_failed_others += 1
                continue
            
            # Compute bounding box from ALL points (encode + decode) for consistent normalization
            all_points = np.vstack([encode_pc, decode_pc])
            bbmin = all_points.min(0)
            bbmax = all_points.max(0)
            
            # Check for degenerate bounding box
            bbox_size = (bbmax - bbmin).max()
            if bbox_size < 1e-6:
                print(f"Skipping {uid}: Degenerate bounding box (size={bbox_size})")
                num_failed_others += 1
                continue
            
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * 0.9 / bbox_size
            
            # Apply same normalization to both point clouds
            encode_pc_normalized = (encode_pc - center) * scale
            decode_pc_normalized = (decode_pc - center) * scale
            
            # Validate normalized coordinates
            if np.isnan(encode_pc_normalized).any() or np.isnan(decode_pc_normalized).any():
                print(f"Skipping {uid}: NaN in normalized coordinates")
                num_failed_others += 1
                continue
            if np.isinf(encode_pc_normalized).any() or np.isinf(decode_pc_normalized).any():
                print(f"Skipping {uid}: Inf in normalized coordinates")
                num_failed_others += 1
                continue
            
            # Check if normalized coordinates are within reasonable range (should be ~[-1, 1])
            encode_max = np.abs(encode_pc_normalized).max()
            decode_max = np.abs(decode_pc_normalized).max()
            if encode_max > 10 or decode_max > 10:
                print(f"Skipping {uid}: Normalized coordinates out of range (encode_max={encode_max:.2f}, decode_max={decode_max:.2f})")
                num_failed_others += 1
                continue

            # Use encode_pc to generate triplane
            batch_encode_pc = torch.from_numpy(encode_pc_normalized).unsqueeze(0).float().to('cuda')

            with torch.no_grad():
                try:
                    # Generate triplane from encode_pc
                    pc_feat = model.pvcnn(batch_encode_pc, batch_encode_pc)
                    planes = model.triplane_transformer(pc_feat)
                    sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
                    
                    # Sample features at decode_pc points
                    tensor_vertices = torch.from_numpy(decode_pc_normalized).reshape(1, -1, 3).to(torch.float32).cuda()
                    
                    # Validate tensor before sampling
                    if torch.isnan(tensor_vertices).any() or torch.isinf(tensor_vertices).any():
                        print(f"Skipping {uid}: Invalid tensor_vertices after conversion to torch")
                        num_failed_others += 1
                        continue
                    
                    point_feat = sample_triplane_feat(part_planes, tensor_vertices)
                    point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)

                    # Save point features
                    np.save(dest_feat_file, point_feat.astype(np.float16))
                    
                except RuntimeError as e:
                    if "CUDA" in str(e) or "index" in str(e).lower():
                        print(f"Skipping {uid}: CUDA error - {str(e)[:100]}")
                        print(f"  encode shape: {encode_pc.shape}, decode shape: {decode_pc.shape}")
                        print(f"  bbox_size: {bbox_size:.6f}, scale: {scale:.6f}")
                        print(f"  normalized range: [{encode_pc_normalized.min():.3f}, {encode_pc_normalized.max():.3f}]")
                        num_failed_others += 1
                        # Clear CUDA cache to recover from error
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
        
        except zipfile.BadZipFile:
            num_bad_zip += 1
            continue

        except Exception:
            num_failed_others += 1
            continue
    
    print(f"Job {args.job_id} - Number of bad zip files: {num_bad_zip}")
    print(f"Job {args.job_id} - Number of failed others: {num_failed_others}")
    print(f"Job {args.job_id} completed successfully!")


if __name__ == "__main__":
    main()