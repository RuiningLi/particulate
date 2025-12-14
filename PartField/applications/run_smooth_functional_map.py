import os, sys
import numpy as np
import torch
import trimesh
import json

sys.path.append("..")
sys.path.append("../third_party/SmoothFunctionalMaps")
sys.path.append("../third_party/SmoothFunctionalMaps/pyFM")

from partfield.config import default_argument_parser, setup
from pyFM.mesh import TriMesh
from pyFM.spectral import mesh_FM_to_p2p
import DiscreteOpt


def vertex_color_map(vertices):
    min_coord, max_coord = np.min(vertices, axis=0, keepdims=True), np.max(vertices, axis=0, keepdims=True)
    cmap = (vertices - min_coord) / (max_coord - min_coord)
    return cmap


if __name__ == '__main__':
    parser = default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args, freeze=False)

    feature_dir = os.path.join("../exp_results", cfg.result_name)

    all_files = cfg.dataset.all_files
    assert len(all_files) % 2 == 0
    num_pairs = len(all_files) // 2

    device = "cuda"

    output_dir = "../exp_results/correspondence/"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_pairs):
        file0 = all_files[2 * i]
        file1 = all_files[2 * i + 1]

        uid0 = file0.split(".")[-2].replace("/", "_")
        uid1 = file1.split(".")[-2].replace("/", "_")

        mesh0 = trimesh.load(os.path.join(feature_dir, f"input_{uid0}_0.ply"), process=True)
        mesh1 = trimesh.load(os.path.join(feature_dir, f"input_{uid1}_0.ply"), process=True)

        feat0 = np.load(os.path.join(feature_dir, f"part_feat_{uid0}_0_batch.npy"))
        feat1 = np.load(os.path.join(feature_dir, f"part_feat_{uid1}_0_batch.npy"))

        assert mesh0.vertices.shape[0] == feat0.shape[0], "num of vertices should match num of features"
        assert mesh1.vertices.shape[0] == feat1.shape[0], "num of vertices should match num of features"

        th_descr0 = torch.tensor(feat0, device=device, dtype=torch.float32)
        th_descr1 = torch.tensor(feat1, device=device, dtype=torch.float32)

        cdist_01 = torch.cdist(th_descr0, th_descr1, p=2)
        p2p_10_init = cdist_01.argmin(dim=0).cpu().numpy()
        p2p_01_init = cdist_01.argmin(dim=1).cpu().numpy()

        fm_mesh0 = TriMesh(mesh0.vertices, mesh0.faces, area_normalize=True, center=True).process(k=200, intrinsic=True)
        fm_mesh1 = TriMesh(mesh1.vertices, mesh1.faces, area_normalize=True, center=True).process(k=200, intrinsic=True)

        model = DiscreteOpt.SmoothDiscreteOptimization(fm_mesh0, fm_mesh1)
        model.set_params("zoomout_rhm")
        model.opt_params.step = 10
        model.solve_from_p2p(p2p_21=p2p_10_init, p2p_12=p2p_01_init, n_jobs=30, verbose=True)

        p2p_10_FM = mesh_FM_to_p2p(model.FM_12, fm_mesh0, fm_mesh1, use_adj=True)

        color0 = vertex_color_map(mesh0.vertices)
        color1 = color0[p2p_10_FM]
        
        output_mesh0 = trimesh.Trimesh(mesh0.vertices, mesh0.faces, vertex_colors=color0)
        output_mesh1 = trimesh.Trimesh(mesh1.vertices, mesh1.faces, vertex_colors=color1)
        
        output_mesh0.export(os.path.join(output_dir, f"correspondence_{uid0}_{uid1}_0.ply"))
        output_mesh1.export(os.path.join(output_dir, f"correspondence_{uid0}_{uid1}_1.ply"))

