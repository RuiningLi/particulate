import sys, os, fnmatch, re
import argparse

import numpy as np
import matplotlib
from matplotlib import colors as mcolors
import matplotlib.cm
import potpourri3d as pp3d
import igl
from arrgh import arrgh

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_mesh", type=str, required=True, help="The mesh to read from from, mesh file format.")
    parser.add_argument("--input_labels", type=str, required=True, help="The labels, as a text file with one entry per line")
    parser.add_argument("--label_count", type=int, default=-1, help="The number of labels to use for the visualization. If -1, computed as max of given labels.")
    parser.add_argument("--output", type=str, required=True, help="The obj file to write output to")

    args = parser.parse_args()


    # Read the mesh
    V, F = igl.read_triangle_mesh(args.input_mesh)

    # Read the scalar function
    S = np.loadtxt(args.input_labels)

    # Convert integers to scalars on [0,1]
    if args.label_count == -1:
        N_max = np.max(S) + 1
    else:
        N_max = args.label_count
    S = S.astype(np.float32) / max(N_max-1, 1) 

    # Validate and write
    if len(S.shape) != 1 or S.shape[0] != F.shape[0]:
        raise ValueError(f"when scalar_on==faces, the scalar should be a length num-faces numpy array, but it has shape {S.shape[0]} and F={F.shape[0]}")

    S = np.stack((S, np.zeros_like(S)), axis=-1)
    
    pp3d.write_mesh(V, F, args.output, UV_coords=S, UV_type='per-face')


if __name__ == "__main__":
    main()