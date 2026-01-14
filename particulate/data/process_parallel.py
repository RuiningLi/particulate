#!/usr/bin/env python3
"""Processes multiple URDF files in parallel.

This script takes a file pattern matching URDF files and processes them in parallel
using multiple processes.

Args:
    file_pattern (str): Glob pattern to match URDF files
    output_dir (str): Base directory where output will be saved
    max_parts (int): Maximum number of parts to process per URDF
    num_part_combinations (int): Number of part combinations to save
    num_frames_per_part_combination (int): Number of frames per part combination
    num_processes (int): Number of parallel processes to use

Returns:
    None
"""

import argparse
import glob
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import json
import shutil

from particulate.data.process_urdf import process_urdf
# from particulate.data.process_usd import process_usd


def input_path_to_output_dir(
    input_path: str,
    output_dir: str,
) -> str:
    return Path(os.path.join(output_dir, input_path.split("/")[-2]))


def process_single_urdf(args: Tuple[str, str, str, int, int, int, bool, bool]) -> None:
    """Process a single URDF file with the given parameters.
    
    Args:
        args: Tuple containing (input_path, output_dir, max_parts, 
              num_part_combinations, num_frames_per_part_combination)
    """
    (
        input_path, 
        output_dir, 
        max_parts, 
        verbose
    ) = args
    
    # Create output directory for this URDF
    urdf_output_dir = input_path_to_output_dir(input_path, output_dir)
    urdf_output_dir.mkdir(parents=True, exist_ok=True
    )
    print("urdf_output_dir", urdf_output_dir)
    # Process the URDF
    try:
        process_urdf(
            input_path=input_path,
            output_dir=urdf_output_dir,
            max_parts=max_parts,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_single_lightwheel_usd(args: Tuple[str, str, int, bool]) -> None:
    (
        input_path, 
        output_dir, 
        max_parts, 
        verbose
    ) = args
    
    # Create output directory for this URDF
    usd_path = input_path_to_output_dir(input_path, output_dir)
    os.makedirs(usd_path, exist_ok=True)
    
    # Process the URDF
    try:
        process_usd(
            input_usd=input_path,
            output_dir=usd_path,
        )
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process multiple files in parallel")
    parser.add_argument("file_pattern", help="Glob pattern to match files to process")
    parser.add_argument("output_dir", help="Base directory for output")
    parser.add_argument("--dataset", type=str, default="partnet-mobility", choices=["partnet-mobility", "lightwheel"], help="Dataset to process")
    parser.add_argument("--max-parts", type=int, default=128, help="Maximum number of parts per URDF")
    parser.add_argument("--num-processes", type=int, default=cpu_count(), help="Number of parallel processes to use")

    args = parser.parse_args()
    
    if args.dataset == "partnet-mobility":
        # Find all matching URDF files
        files = sorted(glob.glob(args.file_pattern))
        if not files:
            print(f"No URDF files found matching pattern: {args.file_pattern}")
            return
        print(f"Found {len(files)} URDF files to process")

    elif args.dataset == "lightwheel":
        files = sorted(glob.glob(args.file_pattern))
        if not files:
            print(f"No USD files found matching pattern: {args.file_pattern}")
            return
        print(f"Found {len(files)} USD files to process")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare arguments for each process
    process_args = [
        (file, args.output_dir, args.max_parts, False)
        for file in files
    ]
    
    if args.dataset == "partnet-mobility":
        # Process URDFs in parallel
        with Pool(args.num_processes) as pool:
            list(tqdm(
                pool.imap(process_single_urdf, process_args), 
                total=len(process_args), 
                desc="Processing URDFs",
            ))
    elif args.dataset == "lightwheel":
        # Process USDs in parallel
        with Pool(args.num_processes) as pool:
            list(tqdm(
                pool.imap(process_single_lightwheel_usd, process_args), 
                total=len(process_args), 
                desc="Processing USDs",
            ))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

if __name__ == "__main__":
    main()
