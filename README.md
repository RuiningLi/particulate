# Particulate: Feed-Forward 3D Object Articulation

<p align="center">
<a href=""><img src="https://img.shields.io/badge/arXiv-pending-blue" alt="arXiv"></a>
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
conda create -n particulate 
pip install -r requirements.txt
```

## 🚀 Inference

## TODO

- [ ] Release the Lightwheel benchmark & evaluation code.
- [ ] Release training and data preprocessing codes.
