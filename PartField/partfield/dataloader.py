import torch
import boto3
import json
from os import path as osp
# from botocore.config import Config
# from botocore.exceptions import ClientError
import h5py
import io
import numpy as np
import skimage
import trimesh
import os
from scipy.spatial import KDTree
import gc
from plyfile import PlyData

## For remeshing
import mesh2sdf
import tetgen
import vtk
import math
import tempfile

### For mesh processing
import pymeshlab

from partfield.utils import *

#########################
## To handle quad inputs
#########################
def quad_to_triangle_mesh(F):
    """
    Converts a quad-dominant mesh into a pure triangle mesh by splitting quads into two triangles.

    Parameters:
        quad_mesh (trimesh.Trimesh): Input mesh with quad faces.

    Returns:
        trimesh.Trimesh: A new mesh with only triangle faces.
    """
    faces = F

    ### If already a triangle mesh -- skip
    if len(faces[0]) == 3:
        return F

    new_faces = []

    for face in faces:
        if len(face) == 4:  # Quad face
            # Split into two triangles
            new_faces.append([face[0], face[1], face[2]])  # Triangle 1
            new_faces.append([face[0], face[2], face[3]])  # Triangle 2
        else:
            print(f"Warning: Skipping non-triangle/non-quad face {face}")

    new_faces = np.array(new_faces)

    return new_faces
#########################

class Demo_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.data_path = cfg.dataset.data_path
        self.is_pc = cfg.is_pc

        all_files = os.listdir(self.data_path)

        selected = []
        for f in all_files:
            if ".ply" in f and self.is_pc:
                selected.append(f)
            elif (".obj" in f or ".glb" in f or ".off" in f) and not self.is_pc:
                selected.append(f)

        self.data_list = selected
        self.pc_num_pts = 100000

        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name

        print("val dataset len:", len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)

    def load_ply_to_numpy(self, filename):
        """
        Load a PLY file and extract the point cloud as a (N, 3) NumPy array.

        Parameters:
            filename (str): Path to the PLY file.

        Returns:
            numpy.ndarray: Point cloud array of shape (N, 3).
        """
        ply_data = PlyData.read(filename)

        # Extract vertex data
        vertex_data = ply_data["vertex"]
        
        # Convert to NumPy array (x, y, z)
        points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T

        return points

    def get_model(self, ply_file):

        uid = ply_file.split(".")[-2].replace("/", "_")

        ####
        if self.is_pc:
            ply_file_read = os.path.join(self.data_path, ply_file)
            pc = self.load_ply_to_numpy(ply_file_read)

            bbmin = pc.min(0)
            bbmax = pc.max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * 0.9 / (bbmax - bbmin).max()
            pc = (pc - center) * scale

        else:
            obj_path = os.path.join(self.data_path, ply_file)
            mesh = load_mesh_util(obj_path)
            vertices = mesh.vertices
            faces = mesh.faces

            bbmin = vertices.min(0)
            bbmax = vertices.max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * 0.9 / (bbmax - bbmin).max()
            vertices = (vertices - center) * scale
            mesh.vertices = vertices

            ### Make sure it is a triangle mesh -- just convert the quad
            mesh.faces = quad_to_triangle_mesh(faces)

            print("before preprocessing...")
            print(mesh.vertices.shape)
            print(mesh.faces.shape)
            print()

            ### Pre-process mesh
            if self.preprocess_mesh:
                # Create a PyMeshLab mesh directly from vertices and faces
                ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)

                # Create a MeshSet and add your mesh
                ms = pymeshlab.MeshSet()
                ms.add_mesh(ml_mesh, "from_trimesh")

                # Apply filters
                ms.apply_filter('meshing_remove_duplicate_faces')
                ms.apply_filter('meshing_remove_duplicate_vertices')
                percentageMerge = pymeshlab.PercentageValue(0.5)
                ms.apply_filter('meshing_merge_close_vertices', threshold=percentageMerge)
                ms.apply_filter('meshing_remove_unreferenced_vertices')

                # Save or extract mesh
                processed = ms.current_mesh()
                mesh.vertices = processed.vertex_matrix()
                mesh.faces = processed.face_matrix()               

                print("after preprocessing...")
                print(mesh.vertices.shape)
                print(mesh.faces.shape)

            ### Save input
            save_dir = f"exp_results/{self.result_name}"
            os.makedirs(save_dir, exist_ok=True)
            view_id = 0            
            mesh.export(f'{save_dir}/input_{uid}_{view_id}.ply')                


            pc, _ = trimesh.sample.sample_surface(mesh, self.pc_num_pts) 

        result = {
                    'uid': uid
                }

        result['pc'] = torch.tensor(pc, dtype=torch.float32)

        if not self.is_pc:
            result['vertices'] = mesh.vertices
            result['faces'] = mesh.faces

        return result

    def __getitem__(self, index):
        
        gc.collect()

        return self.get_model(self.data_list[index])

##############

###############################
class Demo_Remesh_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.data_path = cfg.dataset.data_path

        all_files = os.listdir(self.data_path)

        selected = []
        for f in all_files:
            if (".obj" in f or ".glb" in f):
                selected.append(f)

        self.data_list = selected
        self.pc_num_pts = 100000

        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name

        print("val dataset len:", len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)


    def get_model(self, ply_file):

        uid = ply_file.split(".")[-2]

        ####
        obj_path = os.path.join(self.data_path, ply_file)
        mesh =  load_mesh_util(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces

        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale
        mesh.vertices = vertices

        ### Pre-process mesh
        if self.preprocess_mesh:
            # Create a PyMeshLab mesh directly from vertices and faces
            ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)

            # Create a MeshSet and add your mesh
            ms = pymeshlab.MeshSet()
            ms.add_mesh(ml_mesh, "from_trimesh")

            # Apply filters
            ms.apply_filter('meshing_remove_duplicate_faces')
            ms.apply_filter('meshing_remove_duplicate_vertices')
            percentageMerge = pymeshlab.PercentageValue(0.5)
            ms.apply_filter('meshing_merge_close_vertices', threshold=percentageMerge)
            ms.apply_filter('meshing_remove_unreferenced_vertices')


            # Save or extract mesh
            processed = ms.current_mesh()
            mesh.vertices = processed.vertex_matrix()
            mesh.faces = processed.face_matrix()               

            print("after preprocessing...")
            print(mesh.vertices.shape)
            print(mesh.faces.shape)

        ### Save input
        save_dir = f"exp_results/{self.result_name}"
        os.makedirs(save_dir, exist_ok=True)
        view_id = 0            
        mesh.export(f'{save_dir}/input_{uid}_{view_id}.ply')   

        try:
            ###### Remesh ######
            size= 256
            level = 2 / size

            sdf = mesh2sdf.core.compute(mesh.vertices, mesh.faces, size)
            # NOTE: the negative value is not reliable if the mesh is not watertight
            udf = np.abs(sdf)
            vertices, faces, _, _ = skimage.measure.marching_cubes(udf, level)

            #### Only use SDF mesh ###
            # new_mesh = trimesh.Trimesh(vertices, faces)
            ##########################

            #### Make tet #####
            components = trimesh.Trimesh(vertices, faces).split(only_watertight=False)
            new_mesh = [] #trimesh.Trimesh()
            if len(components) > 100000:
                raise NotImplementedError
            for i, c in enumerate(components):
                c.fix_normals()
                new_mesh.append(c) #trimesh.util.concatenate(new_mesh, c)
            new_mesh = trimesh.util.concatenate(new_mesh)

            # generate tet mesh
            tet = tetgen.TetGen(new_mesh.vertices, new_mesh.faces)
            tet.tetrahedralize(plc=True, nobisect=1., quality=True, fixedvolume=True, maxvolume=math.sqrt(2) / 12 * (2 / size) ** 3)
            tmp_vtk = tempfile.NamedTemporaryFile(suffix='.vtk', delete=True)
            tet.grid.save(tmp_vtk.name)

            # extract surface mesh from tet mesh
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(tmp_vtk.name)
            reader.Update()
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(reader.GetOutputPort())
            surface_filter.Update()
            polydata = surface_filter.GetOutput()
            writer = vtk.vtkOBJWriter()
            tmp_obj = tempfile.NamedTemporaryFile(suffix='.obj', delete=True)
            writer.SetFileName(tmp_obj.name)
            writer.SetInputData(polydata)
            writer.Update()
            new_mesh =  load_mesh_util(tmp_obj.name)
            ##########################

            new_mesh.vertices = new_mesh.vertices * (2.0 / size) - 1.0  # normalize it to [-1, 1]

            mesh = new_mesh
        ####################

        except:
            print("Error in tet.")
            mesh = mesh 

        pc, _ = trimesh.sample.sample_surface(mesh, self.pc_num_pts) 

        result = {
                    'uid': uid
                }

        result['pc'] = torch.tensor(pc, dtype=torch.float32)
        result['vertices'] = mesh.vertices
        result['faces'] = mesh.faces

        return result

    def __getitem__(self, index):
        
        gc.collect()

        return self.get_model(self.data_list[index])


class Correspondence_Demo_Dataset(Demo_Dataset):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.data_path = cfg.dataset.data_path
        self.is_pc = cfg.is_pc

        self.data_list = cfg.dataset.all_files

        self.pc_num_pts = 100000

        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name

        print("val dataset len:", len(self.data_list))
    