from typing import Optional, Tuple

import numpy as np
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COLORS = [
    (72, 36, 117),
    (33, 145, 140),
    (189, 223, 38),
    (153, 80, 8),
    (12, 12, 242),
    (242, 12, 150),
    (12, 242, 150),
    (12, 150, 242)
]
ARROW_COLOR_REVOLUTE = (255, 0, 0)
ARROW_COLOR_PRISMATIC = (255, 255, 0)


def plot_mesh(mesh):
    verts = mesh.vertices
    faces = getattr(mesh, 'faces', [])
            
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    triangles = verts[faces]
    poly = Poly3DCollection(triangles, facecolors='lightblue', edgecolor='none', alpha=0.5, zorder=1)
    ax.add_collection3d(poly)

    min_vals = verts.min(axis=0)
    max_vals = verts.max(axis=0)
    center = (min_vals + max_vals) / 2
    max_range = (max_vals - min_vals).max() / 2.0
    
    # Set axis limits to ensure equal aspect ratio
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.invert_zaxis()
    ax.invert_xaxis()
    ax.view_init(elev=20, azim=120)
    
    # Draw axes (with zorder to ensure they're visible above the mesh)
    length = max_range * 1.2
    # X axis (Red)
    ax.quiver(center[0], center[1], center[2], length, 0, 0, color='r', label='X', 
              linewidth=2, arrow_length_ratio=0.15, zorder=10)
    # Y axis (Green)
    ax.quiver(center[0], center[1], center[2], 0, length, 0, color='g', label='Y',
              linewidth=2, arrow_length_ratio=0.15, zorder=10)
    # Z axis (Blue)
    ax.quiver(center[0], center[1], center[2], 0, 0, length, color='b', label='Z',
              linewidth=2, arrow_length_ratio=0.15, zorder=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Mesh with Axes (Select Up Direction)")

    ax.set_box_aspect([1,1,1])
    return fig


def create_textured_mesh_parts(mesh_parts, colors=COLORS, tex_res=256):
    # Create a texture map with evenly distributed color blocks
    # Use a horizontal strip layout: texture height = tex_res, width = num_parts * tex_res
    texture_height = block_width = tex_res
    texture_width = len(mesh_parts) * block_width
    texture_array = np.zeros((texture_height, texture_width, 3), dtype=np.uint8)

    for i in range(len(mesh_parts)):
        color_rgb = colors[i % len(colors)][:3]
        x_start = i * block_width
        x_end = (i + 1) * block_width
        texture_array[:, x_start:x_end] = color_rgb
    texture = Image.fromarray(texture_array)

    mesh_parts_colored = []
    for i, mesh_part in enumerate(mesh_parts):
        # Create UV coordinates specifically for this part
        # All faces in this part should point to the same color block
        u_center = (i + 0.5) * block_width / texture_width
        v_center = 0.5
        
        # Create UV coordinates for all vertices in this submesh
        num_part_vertices = len(mesh_part.vertices)
        part_uv_coords = np.full((num_part_vertices, 2), [u_center, v_center], dtype=np.float32)
        mesh_part.visual = trimesh.visual.TextureVisuals(uv=part_uv_coords, image=texture)
        mesh_parts_colored.append(mesh_part)
    
    return mesh_parts_colored


def apply_color_with_texture(mesh: trimesh.Trimesh, color: Tuple, tex_res: int = 16) -> trimesh.Trimesh:
    """
    Apply a solid color to a mesh using UV texture coordinates instead of face colors.
    This ensures compatibility with Blender and other tools that don't support face colors.
    
    Args:
        mesh: The mesh to apply color to
        color: Color as tuple (R, G, B) with values 0-1 or (R, G, B, A) with values 0-255
        tex_res: Resolution of the texture (default: 16x16)
    
    Returns:
        mesh: The mesh with texture applied
    """
    # Normalize color to 0-255 range
    if len(color) >= 3:
        if all(c <= 1.0 for c in color[:3]):
            # Color is in 0-1 range, convert to 0-255
            color_rgb = tuple(int(c * 255) for c in color[:3])
        else:
            # Color is already in 0-255 range
            color_rgb = tuple(int(c) for c in color[:3])
    else:
        raise ValueError("Color must have at least 3 components (R, G, B)")
    
    # Create a solid color texture
    texture_array = np.full((tex_res, tex_res, 3), color_rgb, dtype=np.uint8)
    texture = Image.fromarray(texture_array)
    
    # Create UV coordinates (all pointing to center of texture)
    num_vertices = len(mesh.vertices)
    uv_coords = np.full((num_vertices, 2), 0.5, dtype=np.float32)
    
    # Apply texture to mesh
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=texture)
    
    return mesh


def create_sphere(center: np.ndarray, radius: float, color: Tuple[float, float, float]) -> trimesh.Trimesh:
    """
    Create a sphere mesh.
    """
    sphere = trimesh.creation.icosphere(radius=radius, subdivisions=0)
    sphere.vertices += center
    sphere = apply_color_with_texture(sphere, color)
    return sphere


def create_ring(center, normal, major_radius=0.04, minor_radius=0.006, color=(255, 0, 0), segments=32, tube_segments=16):
    """
    Create a 3D ring (torus) perpendicular to a given direction.
    
    Args:
        center: The center position of the ring (3D point)
        normal: The normal direction of the ring plane (will be normalized)
        major_radius: The radius of the ring from center to tube center
        minor_radius: The radius of the tube itself (ring width)
        color: RGB color tuple (can be 0-1 or 0-255 range)
        segments: Number of segments around the ring
        tube_segments: Number of segments around the tube cross-section
    
    Returns:
        trimesh.Trimesh: The ring mesh
    """
    center = np.array(center)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    
    # Find two perpendicular vectors to the normal
    if abs(normal[2]) < 0.9:
        v1 = np.cross(normal, np.array([0, 0, 1]))
    else:
        v1 = np.cross(normal, np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Generate torus vertices
    vertices = []
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        # Point on the major circle
        circle_point = center + major_radius * (np.cos(theta) * v1 + np.sin(theta) * v2)
        # Direction from center to this point on the major circle
        radial_dir = np.cos(theta) * v1 + np.sin(theta) * v2
        
        for j in range(tube_segments):
            phi = 2 * np.pi * j / tube_segments
            # Point on the tube cross-section
            tube_offset = minor_radius * (np.cos(phi) * radial_dir + np.sin(phi) * normal)
            vertex = circle_point + tube_offset
            vertices.append(vertex)
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    for i in range(segments):
        for j in range(tube_segments):
            # Current vertex indices
            v0 = i * tube_segments + j
            v1 = i * tube_segments + (j + 1) % tube_segments
            v2 = ((i + 1) % segments) * tube_segments + (j + 1) % tube_segments
            v3 = ((i + 1) % segments) * tube_segments + j
            
            # Create two triangles for this quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    
    faces = np.array(faces)
    
    # Create mesh with color using UV texture (compatible with Blender)
    ring_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    ring_mesh = apply_color_with_texture(ring_mesh, color)
    
    return ring_mesh


def create_arrow(
    start_point: np.ndarray,
    end_point: np.ndarray,
    color=(1, 0, 0, 1),
    radius: float = 0.03,
    radius_tip: float = 0.05
) -> trimesh.Trimesh:
    """
    Build a 3-D arrow (cylinder + cone) going from `start_point` to `end_point`.
    """
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("start_point and end_point must be different.")

    # Unit vector in arrow direction
    v_dir = direction / length

    # Heuristic: tip is 10 % of length but never longer than 0.07 m
    tip_h = min(0.1 * length, 0.04)
    body_h = length - tip_h
    if body_h <= 0:                             # extremely short arrow fallback
        tip_h = 0.5 * length
        body_h = length - tip_h

    # Cylinder (body) -- origin on z, height along +z
    cyl = trimesh.creation.cylinder(radius=radius, height=body_h, sections=32)
    cyl.apply_translation([0, 0, body_h / 2])   # base sits at z = 0

    # Cone (tip) -- base at z = 0, apex at z = +tip_h
    cone = trimesh.creation.cone(radius=radius_tip, height=tip_h, sections=32)
    cone.apply_translation([0, 0, body_h])      # base starts where cylinder ends

    # Rotate both meshes from +Z to desired direction
    R = trimesh.geometry.align_vectors([0, 0, 1], v_dir)
    cyl.apply_transform(R)
    cone.apply_transform(R)

    # Translate so tail is at start_point
    cyl.apply_translation(start_point)
    cone.apply_translation(start_point)

    cyl = apply_color_with_texture(cyl, color)
    cone = apply_color_with_texture(cone, color)

    return trimesh.util.concatenate([cyl, cone])


def get_3D_arrow_on_points(
    direction: np.ndarray,
    points: np.ndarray,
    fixed_point: Optional[np.ndarray] = None,
    extension: float = 0.05,
) -> Tuple[float, float]:
    """
    Build a 3-D arrow (cylinder + cone) that encloses `points` along `direction`.
    """
    # ── normalise direction ────────────────────────────────────────────────
    direction = np.asarray(direction, dtype=float)
    if np.linalg.norm(direction) == 0:
        raise ValueError("`direction` must be a non-zero vector.")
    d_hat = direction / np.linalg.norm(direction)

    # ── validate points ───────────────────────────────────────────────────
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("`points` must be of shape (N, 3).")

    # ── choose reference point on axis ────────────────────────────────────
    P0 = (
        np.asarray(fixed_point, dtype=float)
        if fixed_point is not None
        else points.mean(axis=0)
    )

    # ── project points onto axis to find extents ──────────────────────────
    scalars = np.dot(points - P0, d_hat)
    if scalars.shape[0] > 0:
        s_min = scalars.min() - max(extension * (scalars.max() - scalars.min()), 0.1)
        s_max = scalars.max() + max(extension * (scalars.max() - scalars.min()), 0.1)
    else:
        s_min = -0.1
        s_max = 0.1

    start_pt = P0 + s_min * d_hat
    end_pt   = P0 + s_max * d_hat

    return start_pt, end_pt
