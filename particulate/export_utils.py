import numpy as np
import json
import os
import trimesh

from particulate.articulation_utils import (
    compute_part_transforms,
    plucker_to_axis_point
)


def export_animated_glb_file(
    mesh_parts,
    unique_part_ids,
    motion_hierarchy,
    is_part_revolute,
    is_part_prismatic,
    revolute_plucker,
    revolute_range,
    prismatic_axis,
    prismatic_range,
    animation_frames,
    output_path,
    include_axes=False,
    axes_meshes=None
):
    """
    Export an animated GLB file with proper node transformations.
    
    This function creates a GLB file with baked animations where each mesh part is a separate node
    with transformation animations (translation, rotation, scale) that represent the articulation
    motion over time.
    
    Args:
        mesh_parts: List of trimesh objects, one per part
        unique_part_ids: Array of unique part IDs
        motion_hierarchy: List of (parent_id, child_id) tuples defining the kinematic tree
        is_part_revolute: Boolean array indicating if each part has a revolute joint
        is_part_prismatic: Boolean array indicating if each part has a prismatic joint
        revolute_plucker: Plucker coordinates for revolute joint axes
        revolute_range: [low, high] angle limits for revolute joints
        prismatic_axis: Direction vectors for prismatic joints
        prismatic_range: [low, high] displacement limits for prismatic joints
        animation_frames: Number of keyframes in the animation
        output_path: Path to the output animated GLB file
        include_axes: Whether to include axis visualization meshes
        axes_meshes: List of trimesh objects representing axis visualizations (arrows/rings)
    
    The animation interpolates linearly from the low limit (state=0) to high limit (state=1)
    over the specified number of frames at 30 FPS.
    """
    import tempfile
    from pygltflib import GLTF2, Animation, AnimationChannel, AnimationSampler, Accessor, BufferView
    import os
    
    # Step 1: Export base mesh using trimesh (which handles textures/UVs correctly)
    # Create a Scene with all parts and axes
    scene = trimesh.Scene()
    
    # Keep track of part names to find their node indices later
    part_node_names = []
    
    for i, mesh_part in enumerate(mesh_parts):
        # Assign a unique name for this part
        # We use a specific prefix to identify it later
        node_name = f"part_node_{i}"
        part_node_names.append(node_name)
        scene.add_geometry(mesh_part, node_name=node_name)
    
    if include_axes and axes_meshes:
        for i, axis_mesh in enumerate(axes_meshes):
            scene.add_geometry(axis_mesh, node_name=f"axis_node_{i}")
    
    # Export to a temporary file using trimesh
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        scene.export(tmp_path)
        
        # Step 2: Load the GLB using pygltflib
        gltf = GLTF2().load(tmp_path)
        
        # Map node names to node indices
        node_name_to_idx = {}
        if gltf.nodes:
            for i, node in enumerate(gltf.nodes):
                if node.name:
                    node_name_to_idx[node.name] = i
        
        # Step 3: Add animation data
        if not gltf.animations:
            gltf.animations = []
        gltf.animations.append(Animation(channels=[], samplers=[]))
    
        animation_idx = len(gltf.animations) - 1
        
        # Get the current binary buffer
        # Read it from the file directly to ensure we have the correct data
        with open(tmp_path, 'rb') as f:
            # GLB format: 12-byte header, then chunks
            header = f.read(12)
            # Read JSON chunk
            json_chunk_length = int.from_bytes(f.read(4), byteorder='little')
            json_chunk_type = f.read(4)
            json_data = f.read(json_chunk_length)
            # Read binary chunk
            bin_chunk_length = int.from_bytes(f.read(4), byteorder='little')
            bin_chunk_type = f.read(4)
            binary_data = bytearray(f.read(bin_chunk_length))
        
        # Helper function to add binary data to the GLB buffer
        def add_to_binary(data_bytes):
            """Add data to binary blob and return BufferView info."""
            nonlocal binary_data
            
            # Align to 4 bytes
            while len(binary_data) % 4 != 0:
                binary_data.append(0)
            
            start = len(binary_data)
            binary_data.extend(data_bytes)
            
            # Update buffer length in gltf structure
            gltf.buffers[0].byteLength = len(binary_data)
            
            return start, len(data_bytes)
        
        # Step 4: Create animation data
        states = np.linspace(0, 1, animation_frames)
        times = np.linspace(0, animation_frames / 30.0, animation_frames).astype(np.float32)  # 30 FPS
        
        # Add time accessor
        time_bytes = times.tobytes()
        time_start, time_length = add_to_binary(time_bytes)
        time_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=time_start,
            byteLength=time_length
        ))
        
        time_acc_idx = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=time_bv_idx,
            componentType=5126,  # FLOAT
            count=len(times),
            type='SCALAR',
            max=[float(times.max())],
            min=[float(times.min())]
        ))
        
        # For each part, create TRS animation samplers
        for part_idx, part_id in enumerate(unique_part_ids):
            # Find the correct node index for this part
            part_node_name = part_node_names[part_idx]
            target_node_idx = node_name_to_idx.get(part_node_name)
            
            if target_node_idx is None:
                print(f"Warning: Could not find node index for part {part_idx} (name: {part_node_name})")
                continue

            # Compute transforms for all frames
            transforms_over_time = []
            for state in states:
                transforms = compute_part_transforms(
                    unique_part_ids,
                    motion_hierarchy,
                    is_part_revolute,
                    is_part_prismatic,
                    revolute_plucker,
                    revolute_range,
                    prismatic_axis,
                    prismatic_range,
                    state
                )
                transforms_over_time.append(transforms[part_id])
            
            # Decompose transforms into TRS
            translations = []
            rotations = []
            scales = []
            
            for T in transforms_over_time:
                # Extract translation
                translation = T[:3, 3]
                translations.append(translation)
                
                # Extract rotation (convert to quaternion)
                R = T[:3, :3]
                # Compute scale
                scale = np.array([
                    np.linalg.norm(R[:, 0]),
                    np.linalg.norm(R[:, 1]),
                    np.linalg.norm(R[:, 2])
                ])
                scales.append(scale)
                
                # Remove scale from rotation matrix
                R_normalized = R / scale
                
                # Convert rotation matrix to quaternion
                trace = np.trace(R_normalized)
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    w = 0.25 / s
                    x = (R_normalized[2, 1] - R_normalized[1, 2]) * s
                    y = (R_normalized[0, 2] - R_normalized[2, 0]) * s
                    z = (R_normalized[1, 0] - R_normalized[0, 1]) * s
                else:
                    if R_normalized[0, 0] > R_normalized[1, 1] and R_normalized[0, 0] > R_normalized[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + R_normalized[0, 0] - R_normalized[1, 1] - R_normalized[2, 2])
                        w = (R_normalized[2, 1] - R_normalized[1, 2]) / s
                        x = 0.25 * s
                        y = (R_normalized[0, 1] + R_normalized[1, 0]) / s
                        z = (R_normalized[0, 2] + R_normalized[2, 0]) / s
                    elif R_normalized[1, 1] > R_normalized[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + R_normalized[1, 1] - R_normalized[0, 0] - R_normalized[2, 2])
                        w = (R_normalized[0, 2] - R_normalized[2, 0]) / s
                        x = (R_normalized[0, 1] + R_normalized[1, 0]) / s
                        y = 0.25 * s
                        z = (R_normalized[1, 2] + R_normalized[2, 1]) / s
                    else:
                        s = 2.0 * np.sqrt(1.0 + R_normalized[2, 2] - R_normalized[0, 0] - R_normalized[1, 1])
                        w = (R_normalized[1, 0] - R_normalized[0, 1]) / s
                        x = (R_normalized[0, 2] + R_normalized[2, 0]) / s
                        y = (R_normalized[1, 2] + R_normalized[2, 1]) / s
                        z = 0.25 * s
                
                rotations.append([x, y, z, w])
            
            translations = np.array(translations, dtype=np.float32)
            rotations = np.array(rotations, dtype=np.float32)
            scales = np.array(scales, dtype=np.float32)
            
            # Add translation accessor
            trans_bytes = translations.tobytes()
            trans_start, trans_length = add_to_binary(trans_bytes)
            trans_bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(BufferView(
                buffer=0,
                byteOffset=trans_start,
                byteLength=trans_length
            ))
            
            trans_acc_idx = len(gltf.accessors)
            gltf.accessors.append(Accessor(
                bufferView=trans_bv_idx,
                componentType=5126,
                count=len(translations),
                type='VEC3',
                max=translations.max(axis=0).tolist(),
                min=translations.min(axis=0).tolist()
            ))
            
            # Add rotation accessor
            rot_bytes = rotations.tobytes()
            rot_start, rot_length = add_to_binary(rot_bytes)
            rot_bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(BufferView(
                buffer=0,
                byteOffset=rot_start,
                byteLength=rot_length
            ))
            
            rot_acc_idx = len(gltf.accessors)
            gltf.accessors.append(Accessor(
                bufferView=rot_bv_idx,
                componentType=5126,
                count=len(rotations),
                type='VEC4',
                max=rotations.max(axis=0).tolist(),
                min=rotations.min(axis=0).tolist()
            ))
            
            # Add scale accessor
            scale_bytes = scales.tobytes()
            scale_start, scale_length = add_to_binary(scale_bytes)
            scale_bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(BufferView(
                buffer=0,
                byteOffset=scale_start,
                byteLength=scale_length
            ))
            
            scale_acc_idx = len(gltf.accessors)
            gltf.accessors.append(Accessor(
                bufferView=scale_bv_idx,
                componentType=5126,
                count=len(scales),
                type='VEC3',
                max=scales.max(axis=0).tolist(),
                min=scales.min(axis=0).tolist()
            ))
            
            # Create animation samplers and channels
            # Translation sampler
            trans_sampler_idx = len(gltf.animations[animation_idx].samplers)
            gltf.animations[animation_idx].samplers.append(AnimationSampler(
                input=time_acc_idx,
                output=trans_acc_idx,
                interpolation='LINEAR'
            ))
            gltf.animations[animation_idx].channels.append(AnimationChannel(
                sampler=trans_sampler_idx,
                target={'node': target_node_idx, 'path': 'translation'}
            ))
            
            # Rotation sampler
            rot_sampler_idx = len(gltf.animations[animation_idx].samplers)
            gltf.animations[animation_idx].samplers.append(AnimationSampler(
                input=time_acc_idx,
                output=rot_acc_idx,
                interpolation='LINEAR'
            ))
            gltf.animations[animation_idx].channels.append(AnimationChannel(
                sampler=rot_sampler_idx,
                target={'node': target_node_idx, 'path': 'rotation'}
            ))
            
            # Scale sampler
            scale_sampler_idx = len(gltf.animations[animation_idx].samplers)
            gltf.animations[animation_idx].samplers.append(AnimationSampler(
                input=time_acc_idx,
                output=scale_acc_idx,
                interpolation='LINEAR'
            ))
            gltf.animations[animation_idx].channels.append(AnimationChannel(
                sampler=scale_sampler_idx,
                target={'node': target_node_idx, 'path': 'scale'}
            ))
    
        # Step 5: Save the animated GLB with updated binary data
        # We need to manually write the GLB file to ensure our binary_data is used        
        # Helper function to recursively convert non-serializable objects to dicts
        def make_json_serializable(obj):
            """Recursively convert objects to JSON-serializable format."""
            # Handle numpy arrays and scalars
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            # Handle objects with __dict__ (like Attributes)
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, bytes, type)):
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):  # Skip private attributes
                        result[key] = make_json_serializable(value)
                return result
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                # Handle objects with to_dict method
                return make_json_serializable(obj.to_dict())
            else:
                # Return primitive types as-is (str, int, float, bool, None)
                return obj
        
        # Helper function to clean GLTF dict by removing null values and empty arrays
        def clean_gltf_dict(obj):
            """Remove null values and empty arrays to comply with GLTF spec."""
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    cleaned_value = clean_gltf_dict(value)
                    # Skip null values (GLTF spec: optional fields should be omitted, not null)
                    if cleaned_value is None:
                        continue
                    # Skip empty arrays (GLTF spec: empty arrays should be omitted)
                    if isinstance(cleaned_value, list) and len(cleaned_value) == 0:
                        continue
                    result[key] = cleaned_value
                return result
            elif isinstance(obj, list):
                cleaned_list = [clean_gltf_dict(item) for item in obj]
                # Filter out None values from lists
                return [item for item in cleaned_list if item is not None]
            else:
                return obj
        
        # Helper function to validate and fix mesh primitives
        def validate_mesh_primitives(gltf_dict):
            """Remove invalid accessor indices from mesh primitives."""
            if 'meshes' not in gltf_dict:
                return gltf_dict
            
            num_accessors = len(gltf_dict.get('accessors', []))
            
            for mesh in gltf_dict['meshes']:
                if 'primitives' not in mesh:
                    continue
                for primitive in mesh['primitives']:
                    if 'attributes' not in primitive:
                        continue
                    # Remove invalid attribute references
                    valid_attributes = {}
                    for attr_name, accessor_idx in primitive['attributes'].items():
                        # Only keep attributes with valid accessor indices
                        if (isinstance(accessor_idx, int) and 
                            accessor_idx >= 0 and 
                            accessor_idx < num_accessors):
                            valid_attributes[attr_name] = accessor_idx
                    primitive['attributes'] = valid_attributes
                    
                    # Validate indices accessor if present
                    if 'indices' in primitive:
                        indices_idx = primitive['indices']
                        if not (isinstance(indices_idx, int) and 
                               indices_idx >= 0 and 
                               indices_idx < num_accessors):
                            del primitive['indices']
                    
                    # Validate material index if present
                    if 'material' in primitive:
                        material_idx = primitive['material']
                        num_materials = len(gltf_dict.get('materials', []))
                        if not (isinstance(material_idx, int) and 
                               material_idx >= 0 and 
                               material_idx < num_materials):
                            del primitive['material']
            
            return gltf_dict
        
        # Helper function to validate node references
        def validate_node_references(gltf_dict):
            """Validate and fix node references to other objects."""
            if 'nodes' not in gltf_dict:
                return gltf_dict
            
            num_meshes = len(gltf_dict.get('meshes', []))
            num_cameras = len(gltf_dict.get('cameras', []))
            num_skins = len(gltf_dict.get('skins', []))
            num_nodes = len(gltf_dict['nodes'])
            
            for node in gltf_dict['nodes']:
                # Validate mesh reference
                if 'mesh' in node:
                    mesh_idx = node['mesh']
                    if not (isinstance(mesh_idx, int) and 
                           mesh_idx >= 0 and 
                           mesh_idx < num_meshes):
                        del node['mesh']
                
                # Validate camera reference
                if 'camera' in node:
                    camera_idx = node['camera']
                    if not (isinstance(camera_idx, int) and 
                           camera_idx >= 0 and 
                           camera_idx < num_cameras):
                        del node['camera']
                
                # Validate skin reference
                if 'skin' in node:
                    skin_idx = node['skin']
                    if not (isinstance(skin_idx, int) and 
                           skin_idx >= 0 and 
                           skin_idx < num_skins):
                        del node['skin']
                
                # Validate children references
                if 'children' in node:
                    valid_children = []
                    for child_idx in node['children']:
                        if (isinstance(child_idx, int) and 
                            child_idx >= 0 and 
                            child_idx < num_nodes):
                            valid_children.append(child_idx)
                    if len(valid_children) > 0:
                        node['children'] = valid_children
                    else:
                        del node['children']
            
            return gltf_dict
        
        # Helper function to validate texture and image references
        def validate_texture_references(gltf_dict):
            """Validate and fix texture and image references."""
            num_images = len(gltf_dict.get('images', []))
            num_samplers = len(gltf_dict.get('samplers', []))
            
            # Validate textures
            if 'textures' in gltf_dict:
                for texture in gltf_dict['textures']:
                    # Validate sampler reference
                    if 'sampler' in texture:
                        sampler_idx = texture['sampler']
                        if not (isinstance(sampler_idx, int) and 
                               sampler_idx >= 0 and 
                               sampler_idx < num_samplers):
                            del texture['sampler']
                    
                    # Validate source (image) reference
                    if 'source' in texture:
                        source_idx = texture['source']
                        if not (isinstance(source_idx, int) and 
                               source_idx >= 0 and 
                               source_idx < num_images):
                            del texture['source']
            
            return gltf_dict
        
        # Update JSON to reflect new buffer size
        gltf_dict = gltf.to_dict()
        # Recursively convert all nested objects to be JSON serializable
        gltf_dict = make_json_serializable(gltf_dict)
        # Validate and fix references
        gltf_dict = validate_mesh_primitives(gltf_dict)
        gltf_dict = validate_node_references(gltf_dict)
        gltf_dict = validate_texture_references(gltf_dict)
        # Clean up null values and empty arrays (must be last to remove invalid fields)
        gltf_dict = clean_gltf_dict(gltf_dict)
        
        # Write GLB file manually
        with open(output_path, 'wb') as f:
            # Write GLB header
            # Magic: "glTF"
            f.write(b'glTF')
            # Version: 2
            f.write((2).to_bytes(4, byteorder='little'))
            # Total length (will update later)
            total_length_pos = f.tell()
            f.write((0).to_bytes(4, byteorder='little'))
            
            # Write JSON chunk
            json_str = json.dumps(gltf_dict, separators=(',', ':'))
            json_bytes = json_str.encode('utf-8')
            json_chunk_length = len(json_bytes)
            # Align JSON to 4 bytes
            while json_chunk_length % 4 != 0:
                json_bytes += b' '
                json_chunk_length += 1
            
            f.write(json_chunk_length.to_bytes(4, byteorder='little'))
            f.write(b'JSON')
            f.write(json_bytes)
            
            # Write binary chunk
            # Align binary to 4 bytes
            while len(binary_data) % 4 != 0:
                binary_data.append(0)
            
            bin_chunk_length = len(binary_data)
            f.write(bin_chunk_length.to_bytes(4, byteorder='little'))
            f.write(b'BIN\x00')
            f.write(binary_data)
            
            # Update total length
            total_length = f.tell()
            f.seek(total_length_pos)
            f.write(total_length.to_bytes(4, byteorder='little'))
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def export_urdf(
    mesh_parts,
    unique_part_ids,
    motion_hierarchy,
    is_part_revolute,
    is_part_prismatic,
    revolute_plucker,
    revolute_range,
    prismatic_axis,
    prismatic_range,
    output_path,
    name="robot"
):
    urdf_dir = os.path.dirname(output_path)
    os.makedirs(urdf_dir, exist_ok=True)
    mesh_dir = os.path.abspath(os.path.join(urdf_dir, "meshes"))
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Identify parents and children
    unique_part_ids_set = set(unique_part_ids)
    parent_map = {}
    children_map = {pid: [] for pid in unique_part_ids}
    for p, c in motion_hierarchy:
        # Filter out hierarchy edges where parts don't exist in the mesh
        if p not in unique_part_ids_set or c not in unique_part_ids_set:
            continue
            
        parent_map[c] = p
        if p in children_map:
            children_map[p].append(c)
        else:
            children_map[p] = [c]

    # Find roots
    roots = []
    for pid in unique_part_ids:
        if pid not in parent_map:
            roots.append(pid)
            
    # Determine local frame origins for each link (in World Coordinates)
    link_origins_world = {}
    
    for i, pid in enumerate(unique_part_ids):
        if pid in roots:
            link_origins_world[pid] = np.zeros(3)
        elif is_part_revolute[pid]:
            # Revolute: Origin at point on axis
            axis, point = plucker_to_axis_point(revolute_plucker[pid])
            link_origins_world[pid] = point
        elif is_part_prismatic[pid]:
            # Prismatic: Origin at Centroid of mesh
            link_origins_world[pid] = mesh_parts[i].vertices.mean(axis=0)
        else:
            # Fixed/Other
            link_origins_world[pid] = mesh_parts[i].vertices.mean(axis=0)

    # Prepare URDF string
    urdf_lines = []
    urdf_lines.append(f'<?xml version="1.0"?>')
    urdf_lines.append(f'<robot name="{name}">')
    
    # Process each part
    for i, pid in enumerate(unique_part_ids):
        mesh = mesh_parts[i]
        origin = link_origins_world[pid]
        
        # Save mesh (centered at local origin)
        mesh_local = mesh.copy()
        mesh_local.vertices -= origin
        
        mesh_filename = f"part_{pid}.obj"
        mesh_path = os.path.join(mesh_dir, mesh_filename)
        mesh_local.export(mesh_path)
        
        link_name = f"link_{pid}"
        
        urdf_lines.append(f'  <link name="{link_name}">')
        urdf_lines.append(f'    <visual>')
        urdf_lines.append(f'      <origin xyz="0 0 0" rpy="0 0 0"/>')
        urdf_lines.append(f'      <geometry>')
        urdf_lines.append(f'        <mesh filename="./meshes/{mesh_filename}"/>')
        urdf_lines.append(f'      </geometry>')
        urdf_lines.append(f'      <material name="material_{pid}">')
        urdf_lines.append(f'        <color rgba="0.8 0.8 0.8 1.0"/>')
        urdf_lines.append(f'      </material>')
        urdf_lines.append(f'    </visual>')
        urdf_lines.append(f'    <collision>')
        urdf_lines.append(f'      <origin xyz="0 0 0" rpy="0 0 0"/>')
        urdf_lines.append(f'      <geometry>')
        urdf_lines.append(f'        <mesh filename="./meshes/{mesh_filename}"/>')
        urdf_lines.append(f'      </geometry>')
        urdf_lines.append(f'    </collision>')
        urdf_lines.append(f'    <inertial>')
        urdf_lines.append(f'      <mass value="1.0"/>')
        urdf_lines.append(f'      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>')
        urdf_lines.append(f'    </inertial>')
        urdf_lines.append(f'  </link>')
        
    # Joints
    for pid in unique_part_ids:
        if pid in parent_map:
            parent_pid = parent_map[pid]
            child_pid = pid
            
            joint_name = f"joint_{parent_pid}_{child_pid}"
            parent_link = f"link_{parent_pid}"
            child_link = f"link_{child_pid}"
            
            p_origin = link_origins_world[parent_pid]
            c_origin = link_origins_world[child_pid]
            offset = c_origin - p_origin
            
            if is_part_revolute[pid]:
                j_type = "revolute"
                axis, _ = plucker_to_axis_point(revolute_plucker[pid])
                axis = axis / (np.linalg.norm(axis) + 1e-6)
                lower, upper = revolute_range[pid]
            elif is_part_prismatic[pid]:
                j_type = "prismatic"
                axis = prismatic_axis[pid]
                axis = axis / (np.linalg.norm(axis) + 1e-6)
                lower, upper = prismatic_range[pid]
            else:
                j_type = "fixed"
                axis = [0, 0, 1]
                lower, upper = 0, 0
                
            urdf_lines.append(f'  <joint name="{joint_name}" type="{j_type}">')
            urdf_lines.append(f'    <parent link="{parent_link}"/>')
            urdf_lines.append(f'    <child link="{child_link}"/>')
            urdf_lines.append(f'    <origin xyz="{offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}" rpy="0 0 0"/>')
            if j_type != "fixed":
                urdf_lines.append(f'    <axis xyz="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}"/>')
                urdf_lines.append(f'    <limit lower="{lower:.6f}" upper="{upper:.6f}" effort="1000" velocity="100"/>')
            urdf_lines.append(f'  </joint>')
            
    urdf_lines.append(f'</robot>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(urdf_lines))
        
    print(f"Exported URDF to {output_path}")


def export_mjcf(
    mesh_parts,
    unique_part_ids,
    motion_hierarchy,
    is_part_revolute,
    is_part_prismatic,
    revolute_plucker,
    revolute_range,
    prismatic_axis,
    prismatic_range,
    output_path,
    name="robot"
):
    import os
    
    mjcf_dir = os.path.dirname(output_path)
    os.makedirs(mjcf_dir, exist_ok=True)
    mesh_dir = os.path.join(mjcf_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Identify parents and children
    unique_part_ids_set = set(unique_part_ids)
    parent_map = {}
    children_map = {pid: [] for pid in unique_part_ids}
    for p, c in motion_hierarchy:
        # Filter out hierarchy edges where parts don't exist in the mesh
        if p not in unique_part_ids_set or c not in unique_part_ids_set:
            continue
            
        parent_map[c] = p
        if p in children_map:
            children_map[p].append(c)
        else:
            children_map[p] = [c]

    # Find roots
    roots = []
    for pid in unique_part_ids:
        if pid not in parent_map:
            roots.append(pid)
            
    # Determine local frame origins for each link (in World Coordinates)
    link_origins_world = {}
    
    for i, pid in enumerate(unique_part_ids):
        if pid in roots:
            link_origins_world[pid] = np.zeros(3)
        elif is_part_revolute[pid]:
            # Revolute: Origin at point on axis
            axis, point = plucker_to_axis_point(revolute_plucker[pid])
            link_origins_world[pid] = point
        elif is_part_prismatic[pid]:
            # Prismatic: Origin at Centroid of mesh
            link_origins_world[pid] = mesh_parts[i].vertices.mean(axis=0)
        else:
            # Fixed/Other
            link_origins_world[pid] = mesh_parts[i].vertices.mean(axis=0)

    # Save meshes and prepare assets
    asset_lines = []
    asset_lines.append(f'  <asset>')
    
    for i, pid in enumerate(unique_part_ids):
        mesh = mesh_parts[i]
        origin = link_origins_world[pid]
        
        # Save mesh (centered at local origin)
        mesh_local = mesh.copy()
        mesh_local.vertices -= origin
        
        mesh_filename = f"part_{pid}.obj"
        mesh_path = os.path.join(mesh_dir, mesh_filename)
        mesh_local.export(mesh_path)
        
        asset_lines.append(f'    <mesh name="mesh_part_{pid}" file="meshes/{mesh_filename}"/>')
        
    asset_lines.append(f'  </asset>')
    
    # Recursive function to build body hierarchy
    def build_body_xml(pid, parent_pid=None, indent="    "):
        lines = []
        
        # Calculate position relative to parent
        origin = link_origins_world[pid]
        if parent_pid is not None:
            parent_origin = link_origins_world[parent_pid]
            rel_pos = origin - parent_origin
        else:
            rel_pos = origin # Relative to world (0,0,0)
            
        lines.append(f'{indent}<body name="link_{pid}" pos="{rel_pos[0]:.6f} {rel_pos[1]:.6f} {rel_pos[2]:.6f}">')
        
        # Add geom
        lines.append(f'{indent}  <geom type="mesh" mesh="mesh_part_{pid}" name="visual_{pid}"/>')
        # Optional: Add collision geom (using same mesh for now)
        # lines.append(f'{indent}  <geom type="mesh" mesh="mesh_part_{pid}" name="collision_{pid}" group="3"/>')
        
        # Add joint if not root
        if parent_pid is not None:
            if is_part_revolute[pid]:
                axis, _ = plucker_to_axis_point(revolute_plucker[pid])
                axis = axis / (np.linalg.norm(axis) + 1e-6)
                lower, upper = revolute_range[pid]
                # Convert radians to degrees for MJCF default
                lower_deg = np.degrees(lower)
                upper_deg = np.degrees(upper)
                lines.append(f'{indent}  <joint name="joint_{pid}" type="hinge" axis="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}" range="{lower_deg:.6f} {upper_deg:.6f}" limited="true"/>')
            elif is_part_prismatic[pid]:
                axis = prismatic_axis[pid]
                axis = axis / (np.linalg.norm(axis) + 1e-6)
                lower, upper = prismatic_range[pid]
                lines.append(f'{indent}  <joint name="joint_{pid}" type="slide" axis="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}" range="{lower:.6f} {upper:.6f}" limited="true"/>')
            else:
                # Fixed joint (no joint element needed in MJCF, bodies are fused)
                pass
                
        # Process children
        for child_pid in children_map[pid]:
            lines.extend(build_body_xml(child_pid, pid, indent + "  "))
            
        lines.append(f'{indent}</body>')
        return lines

    # Build full MJCF
    mjcf_lines = []
    mjcf_lines.append(f'<mujoco model="{name}">')
    mjcf_lines.append(f'  <compiler angle="degree"/>') # Explicitly set angle unit
    mjcf_lines.extend(asset_lines)
    mjcf_lines.append(f'  <worldbody>')
    
    # Add floor (optional but good for visualization)
    # mjcf_lines.append(f'    <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>')
    
    for root_pid in roots:
        mjcf_lines.extend(build_body_xml(root_pid, indent="    "))
        
    mjcf_lines.append(f'  </worldbody>')
    mjcf_lines.append(f'</mujoco>')
    
    with open(output_path, 'w') as f:
        f.write("\n".join(mjcf_lines))
        
    print(f"Exported MJCF to {output_path}")
