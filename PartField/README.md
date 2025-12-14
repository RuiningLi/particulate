# PartField: Learning 3D Feature Fields for Part Segmentation and Beyond [ICCV 2025]
**[[Project]](https://research.nvidia.com/labs/toronto-ai/partfield-release/)** **[[PDF]](https://arxiv.org/pdf/2504.11451)**

Minghua Liu*, Mikaela Angelina Uy*, Donglai Xiang, Hao Su, Sanja Fidler, Nicholas Sharp, Jun Gao



## Overview
![Alt text](assets/teaser.png)

PartField is a feedforward model that predicts part-based feature fields for 3D shapes. Our learned features can be clustered to yield a high-quality part decomposition, outperforming the latest open-world 3D part segmentation approaches in both quality and speed. PartField can be applied to a wide variety of inputs in terms of modality, semantic class, and style. The learned feature field exhibits consistency across shapes, enabling applications such as cosegmentation, interactive selection, and correspondence.

## Table of Contents

- [Pretrained Model](#pretrained-model)
- [Environment Setup](#environment-setup)
- [TLDR](#tldr)
- [Example Run](#example-run)
- [Interactive Tools and Applications](#interactive-tools-and-applications)
- [Evaluation on PartObjaverse-Tiny](#evaluation-on-partobjaverse-tiny)
- [Discussion](#discussion-clustering-with-messy-mesh-connectivities)
- [Citation](#citation)


## Pretrained Model
```
mkdir model
```
The link to download our pretrained model is here: [Trained on Objaverse](https://huggingface.co/mikaelaangel/partfield-ckpt/blob/main/model_objaverse.ckpt). Due to licensing restrictions, we are unable to release the model that was also trained on PartNet.

## Environment Setup

We use Python 3.10 with PyTorch 2.4 and CUDA 12.4. The environment and required packages can be installed individually as follows:
```
conda create -n partfield python=3.10
conda activate partfield
conda install nvidia/label/cuda-12.4.0::cuda
pip install psutil
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
apt install libx11-6 libgl1 libxrender1
pip install vtk
```

An environment file is also provided and can be used for installation:
```
conda env create -f environment.yml
conda activate partfield
```

## TLDR
1. Input data (`.obj` or `.glb` for meshes, `.ply` for splats) are stored in subfolders under `data/`. You can create a new subfolder and copy your custom files into it.  
2. Extract PartField features by running the script `partfield_inference.py`, passing the arguments `result_name [FEAT_FOL]` and `dataset.data_path [DATA_PATH]`. The output features will be saved in `exp_results/partfield_features/[FEAT_FOL]`.  
3. Segmented parts can be obtained by running the script `run_part_clustering.py`, using the arguments `--root exp/[FEAT_FOL]` and `--dump_dir [PART_OUT_FOL]`. The output segmentations will be saved in `exp_results/clustering/[PART_OUT_FOL]`.  
4. Application demo scripts are available in the `applications/` directory and can be used after extracting PartField features (i.e., after running `partfield_inference.py` on the desired demo data).

## Example Run
### Download Demo Data

#### Mesh Data
We showcase the feasibility of PartField using sample meshes from Objaverse (artist-created) and Trellis3D (AI-generated). Sample data can be downloaded below:
```
sh download_demo_data.sh
```
Downloaded meshes can be found in `data/objaverse_samples/` and `data/trellis_samples/`.

#### Gaussian Splats  
We also demonstrate our approach using Gaussian splatting reconstructions as input. Sample splat reconstruction data from the NeRF dataset can be found [here](https://drive.google.com/drive/folders/1l0njShLq37hn1TovgeF-PVGBBrAdNQnf?usp=sharing). Download the data and place it in the `data/splat_samples/` folder.

### Extract Feature Field 
#### Mesh Data

```
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/objaverse dataset.data_path data/objaverse_samples
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/trellis dataset.data_path data/trellis_samples 
```

#### Point Clouds / Gaussian Splats
```
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/splat dataset.data_path data/splat_samples is_pc True
```

### Part Segmentation
#### Mesh Data

We use agglomerative clustering for part segmentation on mesh inputs.
```
python run_part_clustering.py --root exp_results/partfield_features/objaverse --dump_dir exp_results/clustering/objaverse --source_dir data/objaverse_samples --use_agglo True --max_num_clusters 20 --option 0
```

When the input mesh has multiple connected components or poor connectivity, defining face adjacency by connecting geometrically close faces can yield better results (see discussion below):
```
python run_part_clustering.py --root exp_results/partfield_features/trellis --dump_dir exp_results/clustering/trellis --source_dir data/trellis_samples --use_agglo True --max_num_clusters 20 --option 1 --with_knn True
```

Note that agglomerative clustering does not return a fixed clustering result, but rather a hierarchical part tree, where the root node represents the whole shape and each leaf node corresponds to a single triangle face. You can explore more clustering results by adaptively traversing the tree, such as deciding which part should be further segmented.

#### Point Cloud / Gaussian Splats
We use K-Means clustering for part segmentation on point cloud inputs.
```
python run_part_clustering.py --root exp_results/partfield_features/splat --dump_dir exp_results/clustering/splat --source_dir data/splat_samples --max_num_clusters 20 --is_pc True
```

## Interactive Tools and Applications
We include UI tools to demonstrate various applications of PartField. Set up and try out our demos [here](applications/)!

![Alt text](assets/co-seg.png)

![Alt text](assets/regression_interactive_segmentation_guitars.gif)

## Evaluation on PartObjaverse-Tiny

![Alt text](assets/results_combined_compressed2.gif)

To evaluate all models in PartObjaverse-Tiny, you can download the data [here](https://github.com/Pointcept/SAMPart3D/blob/main/PartObjaverse-Tiny/PartObjaverse-Tiny.md) and run the following commands:
```
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/partobjtiny dataset.data_path data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh  n_point_per_face 2000 n_sample_each 10000
python run_part_clustering.py --root exp_results/partfield_features/partobjtiny/ --dump_dir exp_results/clustering/partobjtiny --source_dir data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh --use_agglo True --max_num_clusters 20 --option 0
```
If an OOM error occurs, you can reduce the number of points sampled per face—for example, by setting `n_point_per_face` to 500.

Evaluation metrics can be obtained by running the command below. The per-category average mIoU reported in the paper is also computed.
```
python compute_metric.py
```
This evaluation code builds on top of the implementation released by [SAMPart3D](https://github.com/Pointcept/SAMPart3D). Users with their own data and corresponding ground truths can easily modify this script to compute their metrics.

## Discussion: Clustering with Messy Mesh Connectivities
<!-- Some meshes can get messy with a lot of connected components, here the connectivity information may not be useful, causing failure cases when using Agglomerative clustering. In these cases, we provide two alternatives, 1) cluster using KMeans. We provide sample code below, or 2) converting the input mesh to a manifold surface mesh.

Sample data download:
```
cd data
mkdir messy_meshes_samples
cd messy_meshes_samples
wget https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-007/00790c705e4c4a1fbc0af9bf5c9e9525.glb
wget https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-132/13cc3ffc69964894a2bc94154aed687f.glb
```

Extract Partfield feature on the original mesh and run KMeans clustering:
```
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/messy_meshes_samples dataset.data_path data/messy_meshes_samples
python run_part_clustering.py --root exp_results/partfield_features/messy_meshes_samples/ --dump_dir exp_results/clustering/messy_meshes_samples_kmeans/ --source_dir data/messy_meshes_samples --max_num_clusters 20
```

Extract convert mesh into a surface manifold, extract Partfield feature and run agglomerative clustering:
```
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/messy_meshes_samples_remesh dataset.data_path data/messy_meshes_samples remesh_demo True 
python run_part_clustering_remesh.py --root exp_results/partfield_features/messy_meshes_samples_remesh --dump_dir exp_results/clustering/messy_meshes_samples_remesh --source_dir data/messy_meshes_samples --use_agglo True --max_num_clusters 20 

python run_part_clustering_remesh.py --root exp_results/partfield_features/trellis_remesh --dump_dir exp_results/clustering/trellis_remesh --source_dir data/trellis_samples --use_agglo True --max_num_clusters 20 
``` -->
 When using agglomerative clustering for part segmentation, an adjacency matrix is passed into the algorithm, which ideally requires the mesh to be a single connected component. However, some meshes can be messy, containing multiple connected components. If the input mesh is not a single connected component, we add pseudo-edges to the adjacency matrix to make it one. By default, we take a simple approach: adding `N-1` pseudo-edges as a chain to connect `N` components together. However, this approach can lead to poor results when the mesh is poorly connected and fragmented.

<img src="assets/messy_meshes_screenshot/option0.png" width="480"/>

```
python run_part_clustering.py --root exp_results/partfield_features/trellis --dump_dir exp_results/clustering/trellis_bad --source_dir data/trellis_samples --use_agglo True --max_num_clusters 20 --option 0
```

When this occurs, we explore different options that can lead to better results:

### 1. Preprocess Input Mesh

We can perform a simple cleanup on the input meshes by removing duplicate vertices and faces, and by merging nearby vertices using `pymeshlab`. This preprocessing step can be enabled via a flag when generating PartField features:

```
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/trellis_preprocess dataset.data_path data/trellis_samples preprocess_mesh True
```

When running agglomerative clustering on a cleaned-up mesh, we observe improved part segmentation:

<img src="assets/messy_meshes_screenshot/preprocess.png" width="480"/>

```
python run_part_clustering.py --root exp_results/partfield_features/trellis_preprocess --dump_dir exp_results/clustering/trellis_preprocess --source_dir data/trellis_samples --use_agglo True --max_num_clusters 20 --option 0
```

### 2. Cluster with KMeans

If modifying the input mesh is not desirable and you prefer to avoid preprocessing, an alternative is to use KMeans clustering, which does not rely on an adjacency matrix.


<img src="assets/messy_meshes_screenshot/kmeans.png" width="480"/>

```
python run_part_clustering.py --root exp_results/partfield_features/trellis --dump_dir exp_results/clustering/trellis_kmeans --source_dir data/trellis_samples --max_num_clusters 20 
```

### 3. MST-based Adjacency Matrix

Instead of simply chaining the connected components of the input mesh, we also explore adding pseudo-edges to the adjacency matrix by constructing a KNN graph using face centroids and computing the minimum spanning tree of that graph.

<img src="assets/messy_meshes_screenshot/option1.png" width="480"/>

```
python run_part_clustering.py --root exp_results/partfield_features/trellis --dump_dir exp_results/clustering/trellis_faceadj --source_dir data/trellis_samples --use_agglo True --max_num_clusters 20 --option 1 --with_knn True
```

<!-- ### Remesh with Marching Cubes (experimental)

We also explore computing the SDF of the input mesh and then running marching cubes, resulting in a surface mesh that is guaranteed to be a single connected component. We then run clustering on the new mesh and map back the segmentation labels to the original mesh by a voting scheme.

<img src="assets/messy_meshes_screenshot/remesh.png" width="480"/>

```
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/trellis_remesh dataset.data_path data/trellis_samples remesh_demo True

python run_part_clustering_remesh.py --root exp_results/partfield_features/trellis_remesh --dump_dir exp_results/clustering/trellis_remesh --source_dir data/trellis_samples --use_agglo True --max_num_clusters 20 
```-->

### More Challenging Meshes!  
The proposed approaches improve results for some meshes, but we find that certain cases still do not produce satisfactory segmentations. We leave these challenges for future work. If you're interested, here are some examples of difficult meshes we encountered:

**Challenging Meshes:**
```
cd data
mkdir challenge_samples
cd challenge_samples
wget https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-007/00790c705e4c4a1fbc0af9bf5c9e9525.glb
wget https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-132/13cc3ffc69964894a2bc94154aed687f.glb
```

## Citation
```
@inproceedings{partfield2025,
      title={PartField: Learning 3D Feature Fields for Part Segmentation and Beyond}, 
      author={Minghua Liu and Mikaela Angelina Uy and Donglai Xiang and Hao Su and Sanja Fidler and Nicholas Sharp and Jun Gao},
      year={2025}
}
```

## References  
PartField borrows code from the following repositories:  
- [OpenLRM](https://github.com/3DTopia/OpenLRM)  
- [PyTorch 3D UNet](https://github.com/wolny/pytorch-3dunet)  
- [PVCNN](https://github.com/mit-han-lab/pvcnn)  
- [SAMPart3D](https://github.com/Pointcept/SAMPart3D) — evaluation script  

Many thanks to the authors for sharing their code!
