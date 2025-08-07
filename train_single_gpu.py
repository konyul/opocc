#!/usr/bin/env python
"""Single GPU training script without distributed training."""

import argparse
import os
import sys
import warnings

import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# Add project to path
sys.path.insert(0, os.path.abspath('.'))

# Import custom modules
from projects.occ_plugin.occupancy.apis import custom_train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
        
    # Initialize distributed training with single process
    # This prevents the SyncBN error
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    
    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    
    # Set device
    torch.cuda.set_device(0)
    
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    
    # Build model
    model = build_model(cfg.model)
    model.init_weights()
    
    # Set CLASSES
    if hasattr(datasets[0], 'CLASSES'):
        model.CLASSES = datasets[0].CLASSES
    
    # Start training
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=None,
        meta=None
    )
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    main()