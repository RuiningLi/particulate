# Data Preprocessing

We provide scripts that preprocess raw data (i.e., 3D articulated assets) in `urdf` or `usd` format into training/evaluation formats.

## URDF

To preprocess a `urdf` asset, use the `process_urdf.py` script under `particulate/data`:

```bash
python -m particulate.data.process_urdf path/to/urdf/file.urdf path/to/output/folder
```

The script will save an OBJ file `original.obj` under the output folder, together with meta files `meta.npz`, `link_axes_plucker.npz`, and `link_range.npz` which store the part and articulation related metadata.

## USD

To preprocess a `usd` asset, use the `process_usd.py` script under `particulate/data`:

```bash
python -m particulate.data.process_usd path/to/usd/file.usd path/to/output/folder
```

The script will save files under the output folder in the same format as the one for `urdf` assets: an OBJ file `original.obj` under the output folder, together with meta files `meta.npz`, `link_axes_plucker.npz`, and `link_range.npz` which store the part and articulation related metadata.

## Training Data

For training, we cache points and relevant metadata for efficient data loading.
To do this for a preprocessed output folder, use the `cache_points.py` script under `particulate/data`:

```bash
python -m particulate.data.cache_points --root path/to/the/preprocessed/folder --output_path path/to/the/output/npz/file.npz --num_points 40000 --ratio_sharp 0.5
```

which caches a training datum containing the following attributes to the output path:

```
points
normals
point_to_bone
point_from_sharp
bone_structure
link_axes_plucker
link_range
```

The arg `--ratio_sharp` dictates the percentage of total points which are sampled from _sharp_ edges.
