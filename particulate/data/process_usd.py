#!/usr/bin/env python3
"""Processes a USD file and generates a normalized mesh and motion info.

This script takes a USD file and:
1. Extracts meshes in their original poses
2. Saves OBJ files (no materials)
3. Computes joint axes and anchors in original pose
4. Saves meta.npz with basic structure info
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Union

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

from particulate.articulation_utils import (
    axis_point_to_plucker,
    shift_axes_plucker
)
from particulate.data_utils import (
    load_obj_raw_preserve
)


# ============================== Utils =========================================

def gf_mat4_to_np(matrix: Gf.Matrix4d) -> np.ndarray:
    """Converts a USD Gf.Matrix4d to a numpy array."""
    return np.array(matrix, dtype=float)


def _apply_transform(matrix_4x4: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Applies a 4x4 transformation matrix to a set of 3D points."""
    # Add homogeneous coordinate (w=1)
    points_homogenous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    
    # Apply transform: P_out = P_in @ M^T
    out_4 = points_homogenous @ matrix_4x4.T
    
    # Perspective division
    out = out_4[:, :3] / out_4[:, 3:]
    return out


def _axis_token_to_vec(token: Any) -> np.ndarray:
    """Converts a USD axis token (X/Y/Z) or vector to a numpy unit vector."""
    if token in (UsdPhysics.Tokens.x, 'X', 'x'):
        return np.array([1.0, 0.0, 0.0])
    if token in (UsdPhysics.Tokens.y, 'Y', 'y'):
        return np.array([0.0, 1.0, 0.0])
    if token in (UsdPhysics.Tokens.z, 'Z', 'z'):
        return np.array([0.0, 0.0, 1.0])
        
    # Handle explicit vector input
    try:
        v = np.array(token, dtype=float).reshape(3)
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
    except Exception:
        pass
        
    return np.array([1.0, 0.0, 0.0])  # Fallback to X axis


def _get_attr(prim: Usd.Prim, name: str, default: Any = None) -> Any:
    """Helper to safely get a USD attribute value."""
    attr = prim.GetAttribute(name)
    if not attr:
        return default
    val = attr.Get()
    return val if val is not None else default


def _quat_to_mat(quaternion: Optional[Union[Gf.Quatf, Gf.Quatd]]) -> np.ndarray:
    """Converts a USD quaternion to a 3x3 rotation matrix.
    
    Returns identity matrix if input is None or zero-length.
    """
    if quaternion is None:
        return np.eye(3)
        
    try:
        real = float(quaternion.GetReal())
        imag = list(quaternion.GetImaginary())
        x, y, z = imag
        
        # Check for zero length quaternion
        v = np.array([x, y, z], dtype=float)
        if np.dot(v, v) < 1e-12:
            return np.eye(3)
            
        w = real
        # Standard quaternion to rotation matrix conversion
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=float)
    except Exception:
        return np.eye(3)


# ========================== Union-Find for Fixed Joint Merging ==============

class UnionFind:
    """Union-Find data structure for merging links connected by fixed joints."""
    
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
    
    def find(self, x):
        """Find the representative of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union two elements, always making the alphabetically smaller one the root."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Always make the alphabetically smaller one the root for determinism
        if rx > ry:
            rx, ry = ry, rx
        self.parent[ry] = rx


# ========================== USD Joint Info =================================

def safe_float_or_default(value: Any, default: float) -> float:
    """Safely converts a value to float, handling None, tuples, NaN/Inf."""
    if value is None:
        return default
    
    # Handle tuple/list case (USD API sometimes returns (value,) tuples)
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            return default
        value = value[0]
    
    try:
        float_val = float(value)
        if np.isnan(float_val) or np.isinf(float_val):
            return default
        return float_val
    except (TypeError, ValueError):
        return default


class PhysicsJoint:
    """Represents a physics joint extracted from USD."""
    def __init__(self,
                 prim: Usd.Prim,
                 jtype: str,
                 parent: Sdf.Path,
                 child: Sdf.Path,
                 axis_world: np.ndarray,
                 anchor_world: np.ndarray,
                 lo: Optional[float] = None,
                 hi: Optional[float] = None):
        self.prim = prim
        self.jtype = jtype          # 'revolute', 'prismatic', or 'fixed'
        self.parent = parent        # Parent link path
        self.child = child          # Child link path
        self.axis_world = axis_world       # Joint axis in world frame (unit vector)
        self.anchor_world = anchor_world   # Joint anchor/origin in world frame
        self.lo = lo                # Lower limit
        self.hi = hi                # Upper limit


def compute_joint_axis_anchor(xf_cache: UsdGeom.XformCache,
                              parent_prim: Usd.Prim,
                              joint_prim: Usd.Prim) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes the joint axis and anchor in world space.
    
    Returns:
        Tuple of (axis_world, anchor_world, axis_norm)
    """
    # 1. Get joint local transform in parent frame
    local_pos0 = _get_attr(joint_prim, 'physics:localPos0', Gf.Vec3d(0, 0, 0))
    local_rot0 = _get_attr(joint_prim, 'physics:localRot0', None)
    R0 = _quat_to_mat(local_rot0)
    
    # 2. Get joint axis (local)
    axis_attr = joint_prim.GetAttribute('physics:axis')
    axis_val = axis_attr.Get() if axis_attr else UsdPhysics.Tokens.z
    axis_local = _axis_token_to_vec(axis_val)

    # 3. Get Parent -> World transform
    parent_to_world = gf_mat4_to_np(xf_cache.GetLocalToWorldTransform(parent_prim)).T
    
    # 4. Compute Anchor in World: ParentWorld * LocalPos
    pos0_array = np.array([[local_pos0[0], local_pos0[1], local_pos0[2]]], dtype=float)
    anchor_world = _apply_transform(parent_to_world, pos0_array)[0]

    # 5. Compute Axis in World: ParentRotation * LocalRotation * AxisLocal
    R_parent = parent_to_world[:3, :3]
    # R0 is the rotation from joint frame to parent frame (or vice versa depending on convention, 
    # but code implies R0 rotates axis_local).
    axis_world = (R_parent @ (R0 @ axis_local.reshape(3, 1))).reshape(3)
    
    norm = np.linalg.norm(axis_world)
    if norm > 0:
        axis_world = axis_world / norm
    else:
        axis_world = np.array([0.0, 0.0, 1.0], float)

    return axis_world, anchor_world, norm


def find_usd_joints(stage: Usd.Stage) -> Tuple[List[PhysicsJoint], Set[Sdf.Path]]:
    """Traverses the stage to find Physics joints and their connected links.
    
    Returns:
        (List of PhysicsJoint objects, Set of link Sdf.Paths)
    """
    joints: List[PhysicsJoint] = []
    links: Set[Sdf.Path] = set()
    xf_cache = UsdGeom.XformCache()  # Cache for default time

    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.Joint):
            continue

        # Extract joint type and verify relationships
        jtype = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            usd_joint = UsdPhysics.RevoluteJoint(prim)
            jtype = 'revolute'
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            usd_joint = UsdPhysics.PrismaticJoint(prim)
            jtype = 'prismatic'
        elif prim.IsA(UsdPhysics.FixedJoint):
            usd_joint = UsdPhysics.FixedJoint(prim)
            jtype = 'fixed'
        else:
            continue

        # Get connected bodies
        rel0 = usd_joint.GetBody0Rel()
        rel1 = usd_joint.GetBody1Rel()
        targets0 = rel0.GetTargets() if rel0 else []
        targets1 = rel1.GetTargets() if rel1 else []

        # We need at least one valid target to process a link
        if not targets0 and not targets1:
            continue
            
        # Register links
        if targets0:
            links.add(targets0[0])
        if targets1:
            links.add(targets1[0])

        # We only process full joints (connecting two bodies) for the joint list
        if not (targets0 and targets1):
            continue

        parent_path = targets0[0]
        child_path = targets1[0]
        
        # Fixed joint handling is simpler
        if jtype == 'fixed':
            joints.append(PhysicsJoint(
                prim, 'fixed', parent_path, child_path,
                np.array([0, 0, 1]), np.array([0, 0, 0])
            ))
            continue

        # Compute axis and anchor for moving joints
        axis_world, anchor_world, _ = compute_joint_axis_anchor(
            xf_cache, stage.GetPrimAtPath(parent_path), prim
        )

        # Extract limits
        lo_attr = usd_joint.GetLowerLimitAttr().Get() if usd_joint.GetLowerLimitAttr() else None
        hi_attr = usd_joint.GetUpperLimitAttr().Get() if usd_joint.GetUpperLimitAttr() else None
        
        if jtype == 'revolute':
            lo = safe_float_or_default(lo_attr, -180.0)
            hi = safe_float_or_default(hi_attr, 180.0)
            # Convert degrees to radians
            lo = lo / 180.0 * np.pi
            hi = hi / 180.0 * np.pi
        else:  # prismatic
            lo = safe_float_or_default(lo_attr, 0.0)
            hi = safe_float_or_default(hi_attr, 0.0)

        # Normalize limits if range is negative
        # (heuristic: if lo is negative and hi is near zero, flip direction)
        if lo < -1e-6 and np.abs(hi) < 1e-6:
            lo, hi, axis_world = 0, -lo, -axis_world

        joints.append(PhysicsJoint(
            prim, jtype, parent_path, child_path,
            axis_world, anchor_world, lo, hi
        ))

    return joints, links


# ========================== OBJ Export ================================

def write_obj_from_stage(stage: Usd.Stage, out_path: str, mesh_paths: List[str]) -> Dict[str, int]:
    """Export specified UsdGeom.Mesh prims to a Wavefront OBJ file.

    - Transforms are baked using UsdGeom.XformCache at default time.
    - Preserves n-gons.
    - Writes v, vt, vn if available.
    
    Returns:
        Dictionary mapping mesh path string to vertex count.
    """
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # OBJ format is 1-based. We track global offsets across all meshes.
    v_offset = 0
    vt_offset = 0
    vn_offset = 0

    def _write_mesh_to_obj(mesh: UsdGeom.Mesh, obj_file):
        nonlocal v_offset, vt_offset, vn_offset
        prim = mesh.GetPrim()
        
        # Write Group/Object headers
        obj_file.write(f"o {prim.GetName()}\n")
        obj_file.write(f"g {prim.GetPath()}\n")

        # 1. Geometry (Vertices)
        points = mesh.GetPointsAttr().Get()
        if not points:
            return

        # Transform points to world space
        xf = xcache.GetLocalToWorldTransform(prim)
        # Using list comprehension for transformation
        transformed_points = [xf.Transform(pt) for pt in points]

        for p in transformed_points:
            obj_file.write(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")

        # 2. Normals
        normals_attr = mesh.GetNormalsAttr()
        normals = normals_attr.Get() if normals_attr else None
        normals_interp = mesh.GetNormalsInterpolation() if normals_attr else ""
        
        wrote_normals = False
        if normals:
            # We only support per-vertex normals easily here. 
            # Face-varying normals would require more complex indexing logic matching face vertex indices.
            if normals_interp == UsdGeom.Tokens.vertex and len(normals) == len(points):
                for n in normals:
                    nvec = Gf.Vec3f(n).GetNormalized()
                    obj_file.write(f"vn {nvec[0]:.9g} {nvec[1]:.9g} {nvec[2]:.9g}\n")
                wrote_normals = True

        # 3. UVs (Texture Coordinates)
        primvars = UsdGeom.PrimvarsAPI(prim)
        st_primvar = primvars.GetPrimvar("st")
        uvs = None
        st_interp = None
        
        if st_primvar and st_primvar.HasValue():
            vals = st_primvar.GetAttr().Get()
            if vals:
                uvs = vals
                st_interp = st_primvar.GetInterpolation()
                for uv in uvs:
                    obj_file.write(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")

        # 4. Faces
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        
        current_idx = 0
        for nverts in face_vertex_counts:
            # Get 1-based global vertex indices for this face
            face_vids = [v_offset + face_vertex_indices[current_idx + i] + 1 for i in range(nverts)]
            current_idx += nverts

            # Build face string parts
            parts = []
            
            # Case A: UVs are face-varying (common in USD)
            if uvs and st_interp == UsdGeom.Tokens.faceVarying:
                # We assume UV indices match the order of vertices in the face
                # (1-based global UV indices)
                face_uv_ids = [vt_offset + i + 1 for i in range(current_idx - nverts, current_idx)]
                
                if wrote_normals:
                    # Format: v/vt/vn
                    # Note: We assume vn index matches v index (vn_offset + local_v_index)
                    for vi, ti in zip(face_vids, face_uv_ids):
                         vn_idx = vn_offset + (vi - v_offset)
                         parts.append(f"{vi}/{ti}/{vn_idx}")
                else:
                    # Format: v/vt
                    parts = [f"{vi}/{ti}" for vi, ti in zip(face_vids, face_uv_ids)]
            
            # Case B: Normals but no UVs (or UVs not face-varying)
            elif wrote_normals:
                 # Format: v//vn
                 for vi in face_vids:
                     vn_idx = vn_offset + (vi - v_offset)
                     parts.append(f"{vi}//{vn_idx}")
            
            # Case C: Geometry only
            else:
                # Format: v
                parts = [str(vi) for vi in face_vids]

            obj_file.write(f"f {' '.join(parts)}\n")

        # Update global offsets
        v_offset += len(points)
        if uvs:
            vt_offset += len(uvs)
        if wrote_normals:
            vn_offset += len(points)

    # Ensure schema initialization
    UsdGeom.PointBased

    mesh_to_vert_count: Dict[str, int] = {}
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# OBJ exported from USD\n")
        
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and prim.GetPath().pathString in mesh_paths:
                prev_v_offset = v_offset
                _write_mesh_to_obj(UsdGeom.Mesh(prim), f)
                mesh_to_vert_count[prim.GetPath().pathString] = v_offset - prev_v_offset

    return mesh_to_vert_count


# ============================= Main pipeline ==================================

def process_usd(input_usd: str, output_dir: str):
    """Main processing function to extract mesh and joint info."""
    stage = Usd.Stage.Open(input_usd)
    if not stage:
        print(f"Failed to open USD: {input_usd}")
        return

    # --- 1. Structure Analysis ---
    joints, link_paths = find_usd_joints(stage)
    if not link_paths:
        print("No links found in USD file")
        return
    if len(link_paths) == 1:
        print("Only one link found in USD file")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Get all link names
    link_names = [path.pathString for path in link_paths]
    
    # --- 1b. Merge links connected by fixed joints ---
    # Create Union-Find and merge links connected by fixed joints
    uf = UnionFind(link_names)
    for j in joints:
        if j.jtype == 'fixed':
            parent_name = str(j.parent)
            child_name = str(j.child)
            if parent_name in uf.parent and child_name in uf.parent:
                uf.union(parent_name, child_name)
    
    # Get unique representatives (merged links) in sorted order
    merged_link_names = sorted(set(uf.find(name) for name in link_names))
    merged_link_index_map = {name: i for i, name in enumerate(merged_link_names)}
    
    # Helper function to get merged index for any original link name
    def get_merged_index(link_name: str) -> int:
        return merged_link_index_map[uf.find(link_name)]

    # --- 2. Mesh Collection ---
    # Map each merged link to its visual meshes (from all original links in the group)
    link_to_mesh_paths = {name: [] for name in merged_link_names}
    for link_path in link_paths:
        link_name = link_path.pathString
        merged_name = uf.find(link_name)
        # Find all meshes under this link
        for descendant in Usd.PrimRange(stage.GetPrimAtPath(link_path)):
            if descendant.IsA(UsdGeom.Mesh):
                link_to_mesh_paths[merged_name].append(descendant.GetPath().pathString)

    all_mesh_paths = [m for meshes in link_to_mesh_paths.values() for m in meshes]

    # --- 3. Export Raw OBJ ---
    combined_orig_path = Path(output_dir) / "original.obj"
    mesh_to_vert_count = write_obj_from_stage(stage, str(combined_orig_path), all_mesh_paths)
    
    # --- 4. Build Metadata (Bone Association) ---
    vert_to_link_indices = []
    
    # Create the vertex-to-bone mapping
    # Note: iterating mesh_to_vert_count relies on insertion order from write_obj_from_stage,
    # which matches the Traversal order.
    for mesh_path, vert_count in mesh_to_vert_count.items():
        # Find which merged link this mesh belongs to
        parent_link = next(link for link, meshes in link_to_mesh_paths.items() if mesh_path in meshes)
        link_idx = merged_link_index_map[parent_link]  # Already a merged link name
        vert_to_link_indices.extend([link_idx] * vert_count)

    vert_to_bone_array = np.array(vert_to_link_indices, dtype=np.int16)

    # Build hierarchy (parent-child indices) - skip fixed joints, use merged indices
    link_hierarchy_set = set()
    for j in joints:
        if j.jtype == 'fixed':
            continue  # Fixed joints are merged, skip them
        parent_idx = get_merged_index(str(j.parent))
        child_idx = get_merged_index(str(j.child))
        # Skip if both map to the same merged link (shouldn't happen for non-fixed joints)
        if parent_idx != child_idx:
            link_hierarchy_set.add((parent_idx, child_idx))
    link_hierarchy = sorted(link_hierarchy_set)  # Sort for determinism

    np.savez(
        Path(output_dir) / "meta.npz",
        vert_to_bone=vert_to_bone_array,
        bone_structure=np.array(link_hierarchy, dtype=np.int8)
    )

    # --- 5. Compute Joint Motion Info (Original Scale) ---
    # Initialize containers using merged link indices
    link_axes_plucker = {str(i): np.zeros(12) for i in range(len(merged_link_names))}
    link_range = {str(i): np.zeros(4) for i in range(len(merged_link_names))}
    
    for j in joints:
        if j.jtype == 'fixed':
            continue  # Fixed joints have no motion
        
        # Use merged index for the child
        child_idx = get_merged_index(str(j.child))
        idx_str = str(child_idx)
        
        anchor = j.anchor_world
        axis = j.axis_world
        
        if j.jtype == 'revolute':
            link_axes_plucker[idx_str][:6] = axis_point_to_plucker(axis, anchor)
            link_range[idx_str][:2] = [j.lo, j.hi]
        
        elif j.jtype == 'prismatic':
            link_axes_plucker[idx_str][6:] = axis_point_to_plucker(axis, anchor)
            link_range[idx_str][2:] = [j.lo, j.hi]

    # --- 6. Normalization ---
    # Load the raw mesh we just wrote to compute normalization parameters
    all_verts, all_faces = load_obj_raw_preserve(combined_orig_path)
    x_min, y_min, z_min = all_verts[:, 0].min(), all_verts[:, 1].min(), all_verts[:, 2].min()
    x_max, y_max, z_max = all_verts[:, 0].max(), all_verts[:, 1].max(), all_verts[:, 2].max()
    x_c, y_c, z_c = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5
    scale = 1.0 / max([x_max - x_min, y_max - y_min, z_max - z_min])
    all_verts = (all_verts - np.array([x_c, y_c, z_c])) * scale

    # Apply normalization to motion parameters
    for link_name in link_range.keys():
        # Scale only affects prismatic limits (translation)
        link_range[link_name][2:] *= scale

    for link_name in link_axes_plucker.keys():
        # Shift/Scale revolute axes
        if not np.all(link_axes_plucker[link_name][:3] == 0):
            link_axes_plucker[link_name][:6] = shift_axes_plucker(
                link_axes_plucker[link_name][:6], x_c, y_c, z_c, scale
            )
        # Shift/Scale prismatic axes
        if not np.all(link_axes_plucker[link_name][6:9] == 0):
            link_axes_plucker[link_name][6:] = shift_axes_plucker(
                link_axes_plucker[link_name][6:], x_c, y_c, z_c, scale
            )

    # --- 7. Save Final Normalized Mesh ---
    with open(combined_orig_path, 'w') as f:
        for v in all_verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces (1-based)
        for face in all_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # --- 8. Save Motion Data ---
    link_axes_plucker_path = Path(output_dir) / "link_axes_plucker.npz"
    link_range_path = Path(output_dir) / "link_range.npz"
    
    # Check for NaNs before saving
    for k, v in link_range.items():
        if np.isnan(v).any():
            raise ValueError(f"NaN detected in link_range for link index {k}")

    np.savez(link_axes_plucker_path, **link_axes_plucker)
    np.savez(link_range_path, **link_range)


def main():
    parser = argparse.ArgumentParser(description="Process USD to normalized mesh and motion info.")
    parser.add_argument("input", type=str, help="Input USD file path")
    parser.add_argument("output_dir", type=str, help="Directory to save output files")
    args = parser.parse_args()

    process_usd(args.input, args.output_dir)


if __name__ == "__main__":
    main()
