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

## Evaluation Data 

For evaluation, we transform the output into format:

```json
{
    "xyz": float32[N, 3],
    "part_ids": int32[N],
    "motion_hierarchy": List[Tuple[int, int]],
    "is_part_revolute": bool[P],
    "is_part_prismatic": bool[P],
    "revolute_plucker": float32[P, 6],
    "revolute_range": float32[P, 2],
    "prismatic_axis": float32[P, 3],
    "prismatic_range": float32[P, 2]
}
```

Each is cached as an individual `{Asset_name}.npz` 

The example of the file structure is:
```
|
|-- result_dir
|   |-- Asset_name1.npz
|   |-- Asset_name2.npz
|   ...
```

We provide an option for [sampling points on input meshes](#resample).

#### The meta file fields in detail
<details>
<summary>Show details</summary>

- For parts that are not of a given joint type, the corresponding axis/range entries are zeros.
- P denotes the number of parts; N denotes the number of points.

#### Detailed description:
- xyz: point cloud of the object in normalized coordinates $[-0.5, 0.5]^3$. 
- part_ids: length-N array mapping each point to its part index in [0, P-1]. P is the number of distinct parts. 
- motion_hierarchy: motion tree describing parent→child links between parts. It can be provided as a list of (parent_id, child_id) tuples.
- is_part_revolute: boolean flags per part indicating revolute joints.
- is_part_prismatic: boolean flags per part indicating prismatic joints.
- revolute_plucker: per-part 6D Plücker line representation of the revolute joint axis in the object frame.
- revolute_range: per-part [min, max] angle limits in radians for revolute joints. 
- prismatic_axis: per-part 3D unit vector giving the translation axis for prismatic joints.
- prismatic_range: per-part [min, max] translation limits in the same normalized length units as the mesh. 



#### Plücker coordinate format 
  - For revolute axes, we represent a 3D line (joint axis) with 6 floats `[l_x, l_y, l_z, m_x, m_y, m_z]` where:
    - `l` is the direction vector of the axis
    - `m` is the moment vector, defined as `m = p × l` for any point `p` on the line
    - To recover a unit axis direction and one point on the axis (see `data/utils.py:get_axis_point_from_plucker`)

</details>