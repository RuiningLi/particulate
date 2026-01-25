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
```bash
conda create -n particulate python=3.10
conda activate particulate
pip install -r requirements.txt
```

## 🚀 Inference
To use our model to predict the articulated structure of a custom 3D model (alternatively, you can try our [demo](https://huggingface.co/spaces/rayli/particulate) on HuggingFace without local setup):

```bash
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

## 🔎 Evaluation 

To perform quantitative evaluation with our proposed protocol, during inference, enable `--eval` flag to save the results:

```bash
python infer.py --input_mesh /path/to/an/asset/in/the/evaluation/set.obj --eval --output_dir /output/path/for/infer/asset_name/
```

This will save a `pred.obj` and a `pred.npz` under `$output_dir/eval`. Run inference for all assets. 

Then, use the `cache_gt.py` script under `particulate/data` to convert all the preprocessed ground-truth assets(refer to [DATA.md](https://github.com/RuiningLi/particulate/blob/main/DATA.md)) to the same format, serve as ground truth for later evaluation:

```bash
python -m particulate.data.cache_gt --root_dir  /path/to/directory/of/preprocessed/assets/ --output_dir /path/to/save/cached/ground/truths/
```

With the GT and predicted files ready, we can obtain the evaluation results by:

```bash
python evaluate.py --gt_dir /directory/of/all/preprocessed/gt/file/ --result_dir /directory/of/all/prediction/  --output_dir /directory/of/evaluation/output/
```

Where:
- **`--gt_dir`**: directory produced by `python -m particulate.data.cache_gt ...` containing cached GT `.npz` files named `<asset_name>.npz`.
- **`--result_dir`**: root directory that contains your inference outputs for all assets. `evaluate.py` searches for prediction meshes under `**/eval/*.obj` and expects each asset to have an `eval/` folder (e.g. `results/Blender001/eval/pred.obj` + `results/Blender001/eval/pred.npz`).
- **`--output_dir`**: directory where evaluation JSON files will be written (default: `eval_result`).

<details>
<summary>Step-by-step guide to reproduce our results on PartNet-Mobility test set</summary>

Assuming the URDF assets in the test set are located at `$PARTNET_TEST_SET/*/mobility.urdf`, first preprocess the assets:

```bash
mkdir -p "$PARTNET_PROPROCESSED_DIR" && find "$PARTNET_TEST_SET" -mindepth 2 -maxdepth 2 -name 'mobility.urdf' -path "$PARTNET_TEST_SET/*/mobility.urdf" -print0 | xargs -0 -P "$(nproc)" -I{} bash -lc 'urdf="{}"; obj="$(basename "$(dirname "$urdf")")"; python -m particulate.data.process_urdf "$urdf" "$PARTNET_PROPROCESSED_DIR/$obj"'
```

Then, from the preprocessed folders, we sample `N=100000` points uniformly and cache them together with the articulation attributes:

```bash
mkdir -p "$PARTNET_CACHED_DIR" && find "$PARTNET_PROPROCESSED_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P "$(nproc)" -I{} bash -lc 'd="{}"; b="$(basename "$d")"; python -m particulate.data.cache_points --root "$d" --output_path "$PARTNET_CACHED_DIR/$b" --num_points 100000 --ratio_sharp 0 --format eval'
```

Then, run inference on all assets:

```bash
mkdir -p "$PARTNET_INFERENCE_DIR" && python infer.py --input_mesh "$PARTNET_PROPROCESSED_DIR/*/original.obj" --eval --output_dir "$PARTNET_INFERENCE_DIR" --up_dir Z
```

Finally, perform evaluation:

```bash
mkdir -p "$PARTNET_EVAL_DIR" && python evaluate.py --gt_dir "$PARTNET_CACHED_DIR" --result_dir "$PARTNET_INFERENCE_DIR" --output_dir "$PARTNET_EVAL_DIR"
```

</details>


<details>
<summary>Step-by-step guide to reproduce our results on Lightwheel test set</summary>

Assuming the USD assets in the test set are located at `$LIGHTWHEEL_ROOT/{object_identifier}/{object_identifier}.usd`, first preprocess the assets:

```bash
mkdir -p "$LIGHTWHEEL_PROPROCESSED_DIR" && find "$LIGHTWHEEL_ROOT" -mindepth 2 -maxdepth 2 -name '*.usd' -path "$LIGHTWHEEL_ROOT/*/*.usd" -print0 | xargs -0 -P "$(nproc)" -I{} bash -lc 'usd="{}"; obj="$(basename "$(dirname "$usd")")"; python -m particulate.data.process_usd "$usd" "$LIGHTWHEEL_PROPROCESSED_DIR/$obj"'
```

Then, from the preprocessed folders, we sample `N=100000` points uniformly and cache them together with the articulation attributes:

```bash
mkdir -p "$LIGHTWHEEL_CACHED_DIR" && find "$LIGHTWHEEL_PROPROCESSED_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P "$(nproc)" -I{} bash -lc 'd="{}"; b="$(basename "$d")"; python -m particulate.data.cache_points --root "$d" --output_path "$LIGHTWHEEL_CACHED_DIR/$b" --num_points 100000 --ratio_sharp 0 --format eval'
```

Then, run inference on all assets:

```bash
mkdir -p "$LIGHTWHEEL_INFERENCE_DIR" && python infer.py --input_mesh "$LIGHTWHEEL_PROPROCESSED_DIR/*/original.obj" --eval --output_dir "$LIGHTWHEEL_INFERENCE_DIR" --up_dir Z
```

Finally, perform evaluation:

```bash
mkdir -p "$LIGHTWHEEL_EVAL_DIR" && python evaluate.py --gt_dir $LIGHTWHEEL_CACHED_DIR --result_dir $LIGHTWHEEL_INFERENCE_DIR --output_dir $LIGHTWHEEL_EVAL_DIR
```

</details>

## TODO

- [x] Release data preprocessing code.
- [x] Release the Lightwheel benchmark & evaluation code.
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
