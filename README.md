# Particulate: Feed-Forward 3D Object Articulation

<p align="center">
<a href="https://arxiv.org/abs/2512.11798"><img src="https://img.shields.io/badge/arXiv-2512.11798-b31b1b" alt="arXiv"></a>
<a href="https://ruiningli.com/particulate"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/rayli/particulate/'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20-Demo%20-blue'></a>
</p>

![Teaser](assets/teaser-768p.gif)

## 🌟 Overview
Particulate is a feed-forward approach that, given a single static 3D mesh of an everyday object, directly infers **all** attributes of the underlying articulated structure, including its 3D parts, kinematic structure, and motion constraints.
- **Ultra-fast Inference**: our model recovers a fully articulated 3D object with a single forward pass in ~10 seconds.
- **SOTA Performance**: our model significantly outperforms prior methods on the task of 3D articulation estimation.
- **GenAI Compatible**: our model can also accurately infer the articulated structure of AI-generated 3D assets, enabling full-fledged generation of articulated assets from images or texts when combined with an off-the-shelf 3D generator.

## 🔧 Installation
Our implementation is tested on pytorch==2.4.0 with cuda 12.4 on Ubuntu 22.04. 
```
conda create -n particulate python=3.10
conda activate particulate
pip install -r requirements.txt
```

## 🚀 Inference
To use our model to predict the articulated structure of a custom 3D model (alternatively, you can try our [demo](https://huggingface.co/spaces/rayli/particulate) on HuggingFace without local setup):

```
python infer.py --input_mesh ./hunyuan3d-examples/foldingchair.glb
```

The script will automatically download the pre-trained checkpoint from Huggingface.

Extra arguments:
- `up_dir`: The up direction of the input mesh. Our model is trained on 3D models with up direction +Z. To achieve optimal result, it is important to make sure the input mesh follow the same convention. The script will automatically rotate the input model to be +Z up with this argument. You can use the visualization in the [demo](https://huggingface.co/spaces/rayli/particulate) to determine the up direction.
- `num_points`: The number of points to be sampled as input to the network. Note that we uniformly sample 50% of points and sample the remaining 50% from *sharp* edges. Please make sure the number of uniform points is larger than the number of faces in the input mesh.
- `min_part_confidence`: Increasing this value will merge parts that have low confidence scores to other parts. Consider increasing this value if the prediction is over segmented.
- `no_strict`: By default, the prediction will be post-processed to ensure that each articulated part is a union of different connected components in the original mesh (i.e., no connected components are split across parts). If the input mesh does **not** have clean connected components, please specify `--no_strict`.

## 💾 Data Preprocessing
Please refer to [DATA.md](https://github.com/RuiningLi/particulate/blob/main/DATA.md).

## TODO

- [x] Release data preprocessing code.
- [ ] Release the Lightwheel benchmark & evaluation code.
- [ ] Release training code.

## Citation

```bibtex
@article{li2025particulate,
    title   = {Particulate: Feed-Forward 3D Object Articulation},
    author  = {Ruining Li and Yuxin Yao and Chuanxia Zheng and Christian Rupprecht and Joan Lasenby and Shangzhe Wu and Andrea Vedaldi},
    journal = {arXiv preprint arXiv:2512.11798},
    year    = {2025}
}
```
