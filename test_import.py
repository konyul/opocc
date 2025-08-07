#!/usr/bin/env python
"""Test script to check if all modules are imported correctly."""

import sys
sys.path.insert(0, '.')

try:
    # Test mmdet3d imports
    print("Testing mmdet3d imports...")
    import mmdet3d
    print("✓ mmdet3d imported successfully")
    
    # Test plugin imports
    print("\nTesting plugin imports...")
    import projects.occ_plugin
    print("✓ occ_plugin imported successfully")
    
    # Test specific modules
    print("\nTesting specific modules...")
    from projects.occ_plugin.occupancy.dense_heads import OccHead, OccHeadRellis
    print("✓ OccHead imported successfully")
    print("✓ OccHeadRellis imported successfully")
    
    from projects.occ_plugin.datasets import Rellis3DDataset, CustomNuScenesOccDataset
    print("✓ Rellis3DDataset imported successfully")
    print("✓ CustomNuScenesOccDataset imported successfully")
    
    # Test registration
    print("\nTesting model registration...")
    from mmdet.models import HEADS
    print(f"Registered heads: {list(HEADS._module_dict.keys())}")
    
    from mmdet.datasets import DATASETS
    print(f"\nRegistered datasets: {list(DATASETS._module_dict.keys())}")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)