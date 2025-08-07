#!/usr/bin/env python
"""Test simple data loading."""

import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONPATH'] = '.'

import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

# Load config
cfg = Config.fromfile('projects/configs/baselines/simple_rellis3d.py')

# Build dataset
dataset = build_dataset(cfg.data.train)

print(f"Dataset created: {type(dataset)}")
print(f"Dataset length: {len(dataset)}")

# Try to load one sample
try:
    sample = dataset[0]
    print("\nSuccessfully loaded sample!")
    print("Keys:", list(sample.keys()))
    
    if 'points' in sample:
        points = sample['points']
        if hasattr(points, 'data'):
            points = points.data
        print(f"Points shape: {points.shape}")
        
    if 'gt_occ' in sample:
        gt_occ = sample['gt_occ']
        if hasattr(gt_occ, 'data'):
            gt_occ = gt_occ.data
        print(f"GT occupancy shape: {gt_occ.shape}")
        print(f"Unique labels: {torch.unique(gt_occ)}")
        
except Exception as e:
    print(f"\nError loading sample: {e}")
    import traceback
    traceback.print_exc()