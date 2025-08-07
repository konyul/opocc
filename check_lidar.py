#!/usr/bin/env python3
"""Check the structure of LiDAR point cloud files."""

import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python check_lidar.py <lidar_file_path>")
    sys.exit(1)

lidar_file = sys.argv[1]

try:
    # Read the binary file
    points = np.fromfile(lidar_file, dtype=np.float32)
    
    print(f"Loaded LiDAR file: {lidar_file}")
    print(f"Total values: {len(points)}")
    print(f"File size: {len(points) * 4} bytes")
    
    # Try different dimensions
    possible_dims = [3, 4, 5, 6, 7, 8]
    
    for dim in possible_dims:
        if len(points) % dim == 0:
            num_points = len(points) // dim
            print(f"\nIf {dim}D points: {num_points} points")
            
            # Show first few points
            if num_points > 0:
                reshaped = points.reshape(-1, dim)
                print(f"First 3 points:")
                for i in range(min(3, num_points)):
                    print(f"  Point {i}: {reshaped[i]}")
                
                # Check value ranges
                print(f"Value ranges per dimension:")
                for d in range(dim):
                    min_val = reshaped[:, d].min()
                    max_val = reshaped[:, d].max()
                    print(f"  Dim {d}: [{min_val:.3f}, {max_val:.3f}]")
    
except Exception as e:
    print(f"Error loading LiDAR file: {e}")
    import traceback
    traceback.print_exc()