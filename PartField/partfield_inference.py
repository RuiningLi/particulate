from partfield.config import default_argument_parser, setup
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning
import torch
import glob
import os, sys
import numpy as np
import random

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
    parser = default_argument_parser()

    npz_file = "/scratch/shared/beegfs/ruining/data/articulate-3d/points-all-dinov3/7265-combination_000-pos_000.npz"
    datum = np.load(npz_file)
    pc = datum['points']

    args = parser.parse_args()
    cfg = setup(args, freeze=False)
    predict(cfg)
    
if __name__ == '__main__':
    main()