#!/usr/bin/env python3
"""Processes a URDF file and generates two OBJ files.

This script takes a URDF file as input and generates two OBJ files:
1. Original model with original textures
2. Same model with each link colored differently using a distinct color scheme

Args:
    input_path (str): Path to the input URDF file.
    output_dir (str): Directory where the output OBJ files will be saved.

Returns:
    None

Raises:
    FileNotFoundError: If the input URDF file does not exist.
    ValueError: If no meshes are found in the URDF file.
"""

import argparse
import hashlib
import re
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence, Union, Set

import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from urdf_parser_py.urdf import URDF

from particulate.articulation_utils import (
    axis_point_to_plucker,
    shift_axes_plucker
)
from particulate.data_utils import (
    load_obj_raw_preserve
)

_BASE_COLOR = (0.5, 0.5, 0.5, 1.0)
_HIGHLIGHT_COLOR = (1.0, 0.0, 0.0, 1.0)


def add_dummy_effort(urdf_path: str) -> str:
    """Creates a temporary URDF file with dummy effort and velocity values.

    Args:
        urdf_path (str): Path to the original URDF file.

    Returns:
        str: Path to the temporary URDF file with added dummy values.

    Raises:
        FileNotFoundError: If the input URDF file does not exist.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    
    # Remove contact tags that cause warnings
    for link in root.findall('.//link'):
        for contact in link.findall('contact'):
            link.remove(contact)
    
    # Add effort and velocity attributes to all limit elements
    for limit in root.findall('.//limit'):
        if 'effort' not in limit.attrib:
            limit.set('effort', '100.0')
        if 'velocity' not in limit.attrib:
            limit.set('velocity', '1.0')
    
    # Convert relative mesh paths to absolute paths
    for mesh in root.findall('.//mesh'):
        if 'filename' in mesh.attrib:
            filename = mesh.attrib['filename']
            if not os.path.isabs(filename):
                abs_path = os.path.abspath(os.path.join(urdf_dir, filename))
                mesh.set('filename', abs_path)
    
    fd, temp_path = tempfile.mkstemp(suffix='.urdf')
    os.close(fd)
    
    tree.write(temp_path)
    return temp_path


def get_transform_matrix(origin: Any) -> np.ndarray:
    """Converts URDF origin to 4x4 transformation matrix.

    Args:
        origin: URDF origin object containing xyz and rpy attributes.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    if origin is None:
        return np.eye(4)
    
    rot = trimesh.transformations.euler_matrix(
        origin.rpy[0], origin.rpy[1], origin.rpy[2]
    )
    rot[:3, 3] = origin.xyz
    return rot


def load_model_from_path(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".obj":
        return load_obj_raw_preserve(path)
    else:
        mesh = trimesh.load(path, process=False, maintain_order=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        return mesh.vertices, mesh.faces


# ─────────────────────────── Regex helpers (OBJ only) ──────────────────────────
_TX_RE_NEWMTL = re.compile(r"^\s*newmtl\s+(\S+)")
_TX_RE_MAP    = re.compile(r"^\s*map_\w+\s+(.+)$")

# ─────────────────────────── Utility functions ─────────────────────────────────
def _sha1_of_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def _parse_mtl_blocks(mtl_paths: List[Path]) -> Dict[str, List[str]]:
    """Return {material_name: [lines]} for every material in the supplied .mtl list."""
    blocks: Dict[str, List[str]] = {}
    for mtl in mtl_paths:
        if not mtl.exists():
            continue
        with mtl.open() as fh:
            cur_name: str | None = None
            cur_lines: List[str] = []
            for ln in fh:
                if (m := _TX_RE_NEWMTL.match(ln)):
                    if cur_name is not None:
                        blocks[cur_name] = cur_lines
                    cur_name, cur_lines = m.group(1), [ln]
                else:
                    cur_lines.append(ln)
            if cur_name is not None:
                blocks[cur_name] = cur_lines
    return blocks


def _copy_and_rewrite_textures(
    block: List[str],
    obj_dir: Path,
    tex_dir: Path,
    output_path: Path,
    tex_seen: Dict[str, str],
) -> List[str]:
    """
    Copy every `map_*` texture referenced in *block* into *tex_dir*
    (deduplicated by SHA-1) and rewrite the map_* line to use relative path to tex_dir.
    """
    out: List[str] = []
    for ln in block:
        if not (m := _TX_RE_MAP.match(ln)):
            out.append(ln)
            continue

        tex_path = (obj_dir / m.group(1)).resolve()
        if not tex_path.exists():
            out.append(ln)
            continue

        digest = _sha1_of_file(tex_path)
        new_name = f"{digest}{tex_path.suffix.lower()}"
        if digest not in tex_seen:
            shutil.copyfile(tex_path, tex_dir / new_name)
            tex_seen[digest] = new_name
        prefix = ln.split(m.group(1), 1)[0]
        # Use relative path from mtl file location to tex_dir
        rel_path = os.path.relpath(tex_dir, output_path.parent)
        out.append(f"{prefix}{rel_path}/{new_name}\n")
    return out

# ─────────────────────────── Main merge routine ────────────────────────────────
def combine_meshes_and_save(
    verts: Sequence[np.ndarray],
    faces: Sequence[np.ndarray],
    model_paths: Sequence[Union[str, Path]],
    output_base: Union[str, Path],
    texture_dir: Optional[Union[str, Path]] = None,
    overwrite_colors: Optional[
        Sequence[Optional[Tuple[float, float, float, float]]]
    ] = None,
    verbose: bool = True,
) -> None:
    """
    Merge several meshes into a single OBJ/MTL pair (+textures/).

    Parameters
    ----------
    verts, faces, model_paths
        As before – parallel sequences, one entry per source mesh.
    output_base
        Path stem for the merged files (e.g. “…/merged” → “merged.obj/.mtl”).
    overwrite_colors
        *Optional*, same length as the other lists.  For every mesh:
          • None  → keep / copy its original materials and textures  
          • (r, g, b, a) with components in [0, 1] → ignore the mesh's
            original materials/textures and render the whole mesh with that
            solid colour (alpha → `d` in the MTL).

    Notes
    -----
    * OBJ sources keep their vt / vn indices so that UVs & normals stay valid
      even when the material is overridden.
    * When a colour override is requested we **do not copy any textures** for
      that mesh.
    """
    # ------------------------------------------------------------------ sanity
    n = len(model_paths)
    if not (len(verts) == len(faces) == n):
        raise ValueError("verts, faces and model_paths must be the same length")

    if overwrite_colors is not None and len(overwrite_colors) != n:
        raise ValueError("overwrite_colors must match model count (or be None)")

    output_base = Path(output_base)
    out_obj = output_base.with_suffix(".obj")
    out_mtl = output_base.with_suffix(".mtl")
    if texture_dir is None:
        tex_dir = out_obj.parent / "textures"
    else:
        tex_dir = Path(texture_dir)
    tex_dir.mkdir(parents=True, exist_ok=True)

    obj_buf: List[str] = [f"mtllib {out_mtl.name}\n"]
    mtl_buf: List[str] = []
    tex_seen: Dict[str, str] = {}

    v_ofs = vt_ofs = vn_ofs = 0

    # ---------------------------------------------------------------- per mesh
    for mesh_i, (V_np, F_np, src_raw) in enumerate(
        zip(verts, faces, model_paths)
    ):
        src_path = Path(src_raw)
        ext = src_path.suffix.lower()
        color_override = (
            None if overwrite_colors is None else overwrite_colors[mesh_i]
        )

        if V_np.ndim != 2 or V_np.shape[1] != 3:
            raise ValueError("verts entries must be shaped (Ni, 3)")

        # ───────────────────────────── OBJ input ──────────────────────────
        if ext == ".obj":
            lines = src_path.read_text().splitlines()

            vt_lines, vn_lines = [], []
            face_tokens, face_mats = [], []
            mtl_files: List[str] = []
            cur_mat, explicit_usemtl = None, False

            for ln in lines:
                if ln.startswith("mtllib"):
                    mtl_files.extend(ln.split()[1:])
                elif ln.startswith("vt "):
                    vt_lines.append(ln)
                elif ln.startswith("vn "):
                    vn_lines.append(ln)
                elif ln.startswith("usemtl"):
                    cur_mat = ln.split()[1]
                    explicit_usemtl = True
                elif ln.startswith("f "):
                    toks = ln[2:].strip().split()
                    if len(toks) == 3:
                        face_tokens.append(toks)
                    else:
                        face_tokens.append(toks[:3])
                        for i in range(2, len(toks) - 1):
                            face_tokens.append([toks[0], toks[i], toks[i + 1]])
                    for _ in range(len(toks) - 2):
                        face_mats.append(cur_mat)

            if len(face_tokens) != F_np.shape[0]:
                raise ValueError(
                    f"{src_path.name}: faces array has {F_np.shape[0]} rows "
                    f"but OBJ contains {len(face_tokens)} face lines"
                )

            # ---------- choose material strategy ----------
            if color_override is not None:
                # One synthetic material for this mesh
                r, g, b, a = color_override
                mat_name = f"{mesh_i}_override"
                mtl_buf.append(
                    f"newmtl {mat_name}\n"
                    f"Kd {r:.6f} {g:.6f} {b:.6f}\n"
                    f"Ka {r:.6f} {g:.6f} {b:.6f}\n"
                    f"d {a:.6f}\n"
                    f"Ns 10.0\n\n"
                )
                face_mats = [mat_name] * len(face_tokens)
                name_map: Dict[str, str] = {}

            else:
                # Use real materials / textures (previous behaviour)
                mtl_paths = [src_path.parent / n for n in mtl_files]
                mat_blocks = _parse_mtl_blocks(mtl_paths)
                name_map: Dict[str, str] = {}

                if not explicit_usemtl:  # implicit-material case
                    if len(mat_blocks) == 1:
                        sole = next(iter(mat_blocks))
                        mat_name = f"{mesh_i}_{sole}"
                        rewritten = _copy_and_rewrite_textures(
                            mat_blocks[sole], src_path.parent, tex_dir, out_mtl, tex_seen
                        )
                        rewritten[0] = f"newmtl {mat_name}\n"
                        mtl_buf.extend(
                            rewritten
                            if rewritten[-1].endswith("\n")
                            else rewritten + ["\n"]
                        )
                    else:
                        mat_name = f"{mesh_i}_implicit"
                        mtl_buf.append(
                            f"newmtl {mat_name}\nKd 0.8 0.8 0.8\nNs 10\n\n"
                        )
                    face_mats = [mat_name] * len(face_tokens)
                else:
                    used = {m for m in face_mats if m}
                    for old in used:
                        if old not in mat_blocks:
                            continue
                        new = f"{mesh_i}_{old}"
                        name_map[old] = new
                        rewritten = _copy_and_rewrite_textures(
                            mat_blocks[old],
                            src_path.parent,
                            tex_dir,
                            out_mtl,
                            tex_seen,
                        )
                        rewritten[0] = f"newmtl {new}\n"
                        mtl_buf.extend(
                            rewritten
                            if rewritten[-1].endswith("\n")
                            else rewritten + ["\n"]
                        )

            # ---------- write geometry ----------
            obj_buf.extend(
                f"v {vx:.6f} {vy:.6f} {vz:.6f}\n"
                for vx, vy, vz in V_np.astype(float)
            )
            obj_buf.extend(ln + "\n" for ln in vt_lines)
            obj_buf.extend(ln + "\n" for ln in vn_lines)

            last_mat = None
            for k, toks in enumerate(face_tokens):
                mat_here = name_map.get(face_mats[k], face_mats[k])
                if mat_here != last_mat:
                    obj_buf.append(f"usemtl {mat_here}\n")
                    last_mat = mat_here

                tri = []
                for j, tok in enumerate(toks):
                    p = tok.split("/")
                    v_new = F_np[k, j] + 1 + v_ofs
                    vt_new = vn_new = None
                    if len(p) >= 2 and p[1]:
                        vt_new = int(p[1]) + vt_ofs
                    if len(p) == 3 and p[2]:
                        vn_new = int(p[2]) + vn_ofs

                    if vt_new is None and vn_new is None:
                        tri.append(f"{v_new}")
                    elif vt_new is not None and vn_new is None:
                        tri.append(f"{v_new}/{vt_new}")
                    elif vt_new is None and vn_new is not None:
                        tri.append(f"{v_new}//{vn_new}")
                    else:
                        tri.append(f"{v_new}/{vt_new}/{vn_new}")
                obj_buf.append("f " + " ".join(tri) + "\n")

            vt_len, vn_len = len(vt_lines), len(vn_lines)

        # ─────────────────────── other formats (.stl, .ply, …) ───────────────
        else:
            mat_name = (
                f"{mesh_i}_override"
                if color_override is not None
                else f"{mesh_i}_default"
            )
            if color_override is not None:
                r, g, b, a = color_override
            else:  # neutral grey
                r = g = b = 0.8
                a = 1.0
            mtl_buf.append(
                f"newmtl {mat_name}\n"
                f"Kd {r:.6f} {g:.6f} {b:.6f}\n"
                f"Ka {r:.6f} {g:.6f} {b:.6f}\n"
                f"d {a:.6f}\n"
                f"Ns 10.0\n\n"
            )

            obj_buf.extend(
                f"v {vx:.6f} {vy:.6f} {vz:.6f}\n"
                for vx, vy, vz in V_np.astype(float)
            )
            obj_buf.append(f"usemtl {mat_name}\n")
            for tri in F_np:
                obj_buf.append(
                    "f "
                    + " ".join(str(idx + 1 + v_ofs) for idx in tri)
                    + "\n"
                )
            vt_len = vn_len = 0

        # ---------- advance offsets ----------
        v_ofs += V_np.shape[0]
        vt_ofs += vt_len
        vn_ofs += vn_len

    # ----------------------------------------------------------------- write
    out_obj.write_text("".join(obj_buf))
    out_mtl.write_text("".join(mtl_buf))
    if verbose:
        print(
            f"✅  wrote {out_obj} + {out_mtl.name}   "
            f"({len(tex_seen)} unique texture files)"
        )


def load_urdf(urdf_path: str) -> Optional[URDF]:
    """Load the URDF object."""
    temp_urdf_path = add_dummy_effort(urdf_path)
    try:
        robot = URDF.from_xml_file(temp_urdf_path)
        return robot
    finally:
        os.unlink(temp_urdf_path)


def load_mesh_from_urdf(
    urdf_path: str,
    joint_pos: Dict[str, float] = {},
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, Path]],
    List[Tuple[str, str]],
    Set[str],
]:
    """Loads all meshes from URDF file.

    Args:
        urdf_path (str): Path to the URDF file.
        joint_pos (Dict[str, float], optional): A dictionary of joint names and their positions.

    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, Path]], List[Tuple[str, str]]]: A tuple containing:
            1. A list of tuples, each containing:
                - np.ndarray: The mesh vertices
                - np.ndarray: The mesh faces
                - np.ndarray: The transform matrix
                - str: The name of the link
                - Path: The original file path
            2. A list of tuples containing parent-child link relationships
            3. A set of dummy links which do not have any meshes

    Raises:
        FileNotFoundError: If the input URDF file does not exist.
        ValueError: If no meshes are found in the URDF file.
    """
    temp_urdf_path = add_dummy_effort(urdf_path)
    try:
        robot = URDF.from_xml_file(temp_urdf_path)
        all_meshes = []
        link_hierarchy = []
        dummy_links = set()

        # Build joint tree
        joint_tree: Dict[str, List[Any]] = {}
        for joint in robot.joints:
            if joint.parent not in joint_tree:
                joint_tree[joint.parent] = []
            joint_tree[joint.parent].append(joint)
            link_hierarchy.append((joint.parent, joint.child))

        # Build link transforms
        link_transforms: Dict[str, np.ndarray] = {}
        link_axes_plucker: Dict[str, np.ndarray] = {}
        link_range: Dict[str, np.ndarray] = {}
        
        def compute_link_transform(link_name: str, parent_transform: np.ndarray = np.eye(4)) -> None:
            """Computes transform for a link and all its children recursively.

            Args:
                link_name: Name of the link to compute transform for.
                parent_transform: Transform matrix of the parent link.
            """
            child_joints = joint_tree.get(link_name, [])
            
            for joint in child_joints:
                # Fixed transform from parent → joint frame
                origin_tf = get_transform_matrix(joint.origin)

                # ───────────── variable motion per joint type ──────────────
                if joint.type in ("revolute", "continuous"):
                    limit = getattr(joint, "limit", None)
                    limit_tuple = None
                    if joint.type == "continuous" and limit is None:
                        limit_tuple = (0.0, np.pi * 2)
                    elif limit is not None and limit.lower is not None and limit.upper is not None:
                        limit_tuple = (limit.lower, limit.upper)
                    # choose value between limits according to limit_ratio if they exist, otherwise 0 rad
                    if limit_tuple is not None:
                        q = limit_tuple[0] + joint_pos.get(joint.name, 0.0) * (limit_tuple[1] - limit_tuple[0])
                    else:
                        q = 0.0
                    motion_tf = trimesh.transformations.rotation_matrix(
                        q, np.asarray(joint.axis, float)
                    )

                elif joint.type == "prismatic":
                    # translate along the joint axis (metres)
                    if getattr(joint, "limit", None) and \
                       joint.limit.lower is not None and joint.limit.upper is not None:
                        d = joint.limit.lower + joint_pos.get(joint.name, 0.0) * (joint.limit.upper - joint.limit.lower)
                    else:
                        d = 0.0
                    motion_tf = np.eye(4)
                    axis_vec = np.asarray(joint.axis, float)
                    if np.linalg.norm(axis_vec) > 0:
                        motion_tf[:3, 3] = axis_vec / np.linalg.norm(axis_vec) * d

                else:
                    # fixed, planar, floating → no variable part applied
                    motion_tf = np.eye(4)

                # Full joint transform = origin * motion
                joint_transform = origin_tf @ motion_tf
                child_link = next(l for l in robot.links if l.name == joint.child)
                child_transform = parent_transform @ joint_transform
                link_transforms[child_link.name] = child_transform
                if joint.axis is not None:
                    point_homogeneous = parent_transform @ origin_tf @ np.append(joint.axis, 1.0)
                    point = point_homogeneous[:3] / point_homogeneous[3]
                    axis = (parent_transform @ origin_tf @ np.append(joint.axis, 0.0))[:3]
                    if child_link.name not in link_axes_plucker:
                        link_axes_plucker[child_link.name] = np.zeros(12)
                    if joint.type in ("revolute", "continuous"):
                        link_axes_plucker[child_link.name][:6] = axis_point_to_plucker(axis, point)
                    elif joint.type == "prismatic":
                        link_axes_plucker[child_link.name][6:12] = axis_point_to_plucker(axis, point)
                    link_range[child_link.name] = np.zeros(4)
                    if joint.type == "continuous":
                        link_range[child_link.name][:2] = (-np.pi, np.pi)
                    elif joint.type == "revolute":
                        limit = getattr(joint, "limit", None)
                        if limit is not None and limit.lower is not None and limit.upper is not None:
                            link_range[child_link.name][:2] = (limit.lower - q, limit.upper - q)
                    elif joint.type == "prismatic":
                        limit = getattr(joint, "limit", None)
                        if limit is not None and limit.lower is not None and limit.upper is not None:
                            link_range[child_link.name][2:] = (limit.lower - d, limit.upper - d)
                compute_link_transform(child_link.name, child_transform)
        
        # Find root links
        child_links = set(joint.child for joint in robot.joints)
        root_links = [link for link in robot.links if link.name not in child_links]
        
        # Compute transforms starting from root links
        for root_link in root_links:
            link_transforms[root_link.name] = np.eye(4)
            compute_link_transform(root_link.name)
        
        # Process all links and their meshes
        processed_links = set()
        for link in robot.links:
            visual_elements = []
            if hasattr(link, 'visuals') and link.visuals:
                visual_elements.extend(link.visuals)
            if hasattr(link, 'visual') and link.visual:
                visual_elements.append(link.visual)

            if len(visual_elements) == 0:
                dummy_links.add(link.name)
                continue
            
            for visual in visual_elements:
                if visual.geometry and hasattr(visual.geometry, 'filename'):
                    mesh_path = visual.geometry.filename
                    assert os.path.exists(mesh_path), f"Mesh file does not exist: {mesh_path}"
                    
                    try:
                        loaded_verts, loaded_faces = load_model_from_path(Path(mesh_path))
                        
                        link_transform = link_transforms[link.name]
                        
                        if visual.origin:
                            visual_transform = get_transform_matrix(visual.origin)
                            link_transform = link_transform @ visual_transform
                        
                        all_meshes.append((loaded_verts, loaded_faces, link_transform, link.name, Path(mesh_path)))
                        processed_links.add(link.name)
                    except Exception as e:
                        print(f"Failed to load mesh {mesh_path}: {e}")
                        return []
                    
        return all_meshes, link_hierarchy, link_axes_plucker, link_range, dummy_links
    
    finally:
        os.unlink(temp_urdf_path)


def _process_urdf_single_frame(
    input_path: str,
    output_dir: str,
    texture_dir: str,
    meta_path: str,
    joint_pos: Dict[str, float],
    max_parts: int,
    verbose: bool = True,
) -> bool:
    """Processes a URDF file for a single frame."""
    robot = load_urdf(input_path)
    if robot is None:
        print(f"Failed to load URDF file: {input_path}")
        return False
    
    meshes, link_hierarchy, link_axes_plucker, link_range, dummy_links = load_mesh_from_urdf(input_path, joint_pos)

    patterns_to_replace = []
    for p, c in link_hierarchy:
        if c in dummy_links:
            # Find all children of this dummy link
            for p2, c2 in link_hierarchy:
                if p2 == c:
                    patterns_to_replace.append(((p, c), (c, c2), (p, c2)))
    
    for (p1, c1), (p2, c2), (p_new, c_new) in patterns_to_replace:
        if (p1, c1) in link_hierarchy:
            link_hierarchy.remove((p1, c1))
        if (p2, c2) in link_hierarchy:
            link_hierarchy.remove((p2, c2))
        link_hierarchy.append((p_new, c_new))
        if np.all(link_axes_plucker[c1][:3] == 0) and np.all(link_axes_plucker[c_new][6:9] == 0):
            if verbose: print(f"Replacing {c_new} with {c1} prismatic")
            link_axes_plucker[c_new][6:] = link_axes_plucker[c1][6:]
        elif np.all(link_axes_plucker[c1][6:9] == 0) and np.all(link_axes_plucker[c_new][:3] == 0):
            if verbose: print(f"Replacing {c_new} with {c1} revolute")
            link_axes_plucker[c_new][:6] = link_axes_plucker[c1][:6]
        if np.all(link_range[c1][:2] == 0) and np.all(link_range[c_new][2:] == 0):
            if verbose: print(f"Replacing {c_new} with {c1} prismatic")
            link_range[c_new][2:] = link_range[c1][2:]
        elif np.all(link_range[c1][2:] == 0) and np.all(link_range[c_new][:2] == 0):
            if verbose: print(f"Replacing {c_new} with {c1} revolute")
            link_range[c_new][:2] = link_range[c1][:2]

    if not meshes:
        return False

    # Add default material first
    unique_links = sorted(list(set(link_name for _, _, _, link_name, _ in meshes)))

    link_axes_plucker = {str(unique_links.index(k)): v for k, v in link_axes_plucker.items() if k in unique_links}
    link_range = {str(unique_links.index(k)): v for k, v in link_range.items() if k in unique_links}

    if len(unique_links) > max_parts:
        print(f"Skipping {input_path} because it has {len(unique_links)} parts, which is greater than {max_parts}")
        return False
    
    all_verts, all_faces, all_model_paths = [], [], []
    vert_to_link = []
    for verts, faces, transform, link_name, model_path in meshes:
        all_verts.append(trimesh.transform_points(verts, transform))
        all_faces.append(faces)
        all_model_paths.append(model_path)
        vert_to_link.extend([unique_links.index(link_name)] * len(verts))

    if len(list(set(vert_to_link))) <= 1:
        return False

    x_min, y_min, z_min = np.concatenate(all_verts, axis=0).min(axis=0)
    x_max, y_max, z_max = np.concatenate(all_verts, axis=0).max(axis=0)
    x_center, y_center, z_center = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5
    scale = 1.0 / max([x_max - x_min, y_max - y_min, z_max - z_min])
    all_verts = [(v - np.array([x_center, y_center, z_center])) * scale for v in all_verts]
    for link_name in link_range.keys():
        link_range[link_name][2:] *= scale  # Only scale the prismatic range
    
    for link_name in link_axes_plucker.keys():
        if not np.all(link_axes_plucker[link_name][:3] == 0):
            link_axes_plucker[link_name][:6] = shift_axes_plucker(
                link_axes_plucker[link_name][:6], x_center, y_center, z_center, scale
            )
        if not np.all(link_axes_plucker[link_name][6:9] == 0):
            link_axes_plucker[link_name][6:] = shift_axes_plucker(
                link_axes_plucker[link_name][6:], x_center, y_center, z_center, scale
            )

    np.savez(os.path.join(output_dir, "link_axes_plucker.npz"), **link_axes_plucker)
    np.savez(os.path.join(output_dir, "link_range.npz"), **link_range)
    
    if not os.path.exists(meta_path):
        link_hierarchy = [
            (unique_links.index(p), unique_links.index(c)) for p, c in link_hierarchy 
            if p in unique_links and c in unique_links
        ]

        np.savez(
            meta_path,
            vert_to_bone=np.array(vert_to_link, dtype=np.int8),
            bone_structure=np.array(link_hierarchy, dtype=np.int8),
        )
    # Save the original obj, without coloring
    combine_meshes_and_save(
        all_verts, all_faces, all_model_paths, 
        os.path.join(output_dir, "original"), texture_dir=texture_dir, verbose=verbose
    )

    return True


def process_urdf(
    input_path: str, 
    output_dir: str, 
    max_parts: int = 16, 
    verbose: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    robot = load_urdf(input_path)
    if robot is None:
        print(f"Failed to load URDF file: {input_path}")
        return

    joint_names = [joint.name for joint in robot.joints if joint.type in ("revolute", "continuous", "prismatic")]
    joint_pos = {joint_name: 0 for joint_name in joint_names}

    if not _process_urdf_single_frame(
        input_path, 
        output_dir, 
        os.path.join(output_dir, "textures"), 
        os.path.join(output_dir, "meta.npz"), 
        joint_pos, 
        max_parts,
        verbose
    ):
        os.rmdir(output_dir)


def main() -> None:
    """Main function to process command line arguments and run the script."""
    parser = argparse.ArgumentParser(description="Process URDF file and generate two OBJ files")
    parser.add_argument("input", type=str, help="Input URDF file path")
    parser.add_argument("output_dir", type=str, help="Output directory for OBJ files")
    parser.add_argument("--max_parts", type=int, default=64, help="Maximum number of parts to process")
    args = parser.parse_args()

    process_urdf(args.input, args.output_dir, args.max_parts, False)


if __name__ == "__main__":
    main()
