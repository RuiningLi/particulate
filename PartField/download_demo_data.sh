#!/bin/bash
mkdir data
cd data
mkdir objaverse_samples
cd objaverse_samples
wget https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-050/00200996b8f34f55a2dd2f44d316d107.glb
wget https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-042/002e462c8bfa4267a9c9f038c7966f3b.glb
wget https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-046/0c3ca2b32545416f8f1e6f0e87def1a6.glb
wget https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-063/65c6ffa083c6496eb84a0aa3c48d63ad.glb

cd ..
mkdir trellis_samples
cd trellis_samples
wget https://github.com/Trellis3D/trellis3d.github.io/raw/refs/heads/main/assets/scenes/blacksmith/glbs/dwarf.glb
wget https://github.com/Trellis3D/trellis3d.github.io/raw/refs/heads/main/assets/img2/glbs/goblin.glb
wget https://github.com/Trellis3D/trellis3d.github.io/raw/refs/heads/main/assets/img2/glbs/excavator.glb
wget https://github.com/Trellis3D/trellis3d.github.io/raw/refs/heads/main/assets/img2/glbs/elephant.glb
cd ..
cd ..
