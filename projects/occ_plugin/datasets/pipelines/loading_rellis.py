import os
import numpy as np
import torch
import copy
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadOccupancyRellis(object):
    """Load occupancy ground truth for Rellis-3D dataset.
    
    This is a modified version of LoadOccupancy that works with
    Rellis-3D/OCCFusion data structure.
    """
    
    def __init__(self, to_float32=True, use_semantic=False, 
                 occ_path=None, grid_size=[256, 256, 32], 
                 unoccupied=0, pc_range=None, 
                 gt_resize_ratio=1, cal_visible=False,
                 use_vel=False):
        self.to_float32 = to_float32
        self.use_semantic = use_semantic
        self.occ_path = occ_path
        self.cal_visible = cal_visible
        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_vel = use_vel
    
    def __call__(self, results):
        # For Rellis-3D, we directly use the occ_label_path
        occ_label_path = results.get('occ_label_path', '')
        
        if not os.path.exists(occ_label_path):
            # Try to construct path from sample info
            sample_id = results.get('sample_id', '')
            if sample_id:
                # Try different possible paths
                possible_paths = [
                    os.path.join(self.occ_path, f"{sample_id}_occupancy.npy"),
                    os.path.join(self.occ_path, f"{sample_id}.npy"),
                    occ_label_path  # Original path
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        occ_label_path = path
                        break
        
        if not os.path.exists(occ_label_path):
            # Create empty occupancy if file not found
            print(f"Warning: Occupancy file not found: {occ_label_path}")
            # Create empty occupancy grid
            gt_occ = np.ones(self.grid_size, dtype=np.uint8) * 255  # All unknown
            results['gt_occ'] = gt_occ
            return results
        
        # Load occupancy data based on file extension
        if occ_label_path.endswith('.label'):
            # Raw binary format for .label files
            occ_data = np.fromfile(occ_label_path, dtype=np.uint8)
        else:
            # NumPy format for .npy files
            try:
                occ_data = np.load(occ_label_path)
            except ValueError:
                occ_data = np.load(occ_label_path, allow_pickle=True)
        
        # Handle different data formats
        if occ_data.ndim == 1:
            # Flat array - reshape to grid
            expected_size = np.prod(self.grid_size)
            if occ_data.size == expected_size:
                gt_occ = occ_data.reshape(self.grid_size)
            else:
                print(f"Warning: Occupancy size mismatch. Expected {expected_size}, got {occ_data.size}")
                gt_occ = np.ones(self.grid_size, dtype=np.uint8) * 255
        elif occ_data.ndim == 3:
            # Already in grid format
            gt_occ = occ_data
        elif occ_data.ndim == 2:
            # [N, 4] format with [x, y, z, label]
            gt_occ = np.ones(self.grid_size, dtype=np.uint8) * 255  # Initialize as unknown
            coords = occ_data[:, :3].astype(np.int32)
            labels = occ_data[:, 3].astype(np.uint8)
            
            # Filter valid coordinates
            valid = (coords >= 0).all(axis=1) & (coords < self.grid_size).all(axis=1)
            coords = coords[valid]
            labels = labels[valid]
            
            # Fill grid
            gt_occ[coords[:, 0], coords[:, 1], coords[:, 2]] = labels
        else:
            print(f"Warning: Unknown occupancy format with shape {occ_data.shape}")
            gt_occ = np.ones(self.grid_size, dtype=np.uint8) * 255
        
        # Ensure correct data type
        gt_occ = gt_occ.astype(np.uint8)
        
        # Map class labels if needed (Rellis-3D: 0=empty, 1=traversable, 2=non-traversable)
        # No remapping needed if already in correct format
        
        # Resize if needed
        if gt_occ.shape != tuple(self.grid_size):
            print(f"Warning: Resizing occupancy from {gt_occ.shape} to {self.grid_size}")
            import torch.nn.functional as F
            import torch
            # Convert to tensor, add batch and channel dims
            gt_occ_tensor = torch.from_numpy(gt_occ).float().unsqueeze(0).unsqueeze(0)
            # Resize
            target_size = tuple(self.grid_size)
            gt_occ_resized = F.interpolate(gt_occ_tensor, size=target_size, mode='nearest')
            # Convert back to numpy
            gt_occ = gt_occ_resized.squeeze().numpy().astype(np.uint8)
        
        # Add to results
        results['gt_occ'] = gt_occ
        
        # Add other required fields
        if 'bda_mat' not in results:
            results['bda_mat'] = np.eye(4, dtype=np.float32)
        
        return results
    
    def voxel2world(self, voxel):
        """Convert voxel coordinates to world coordinates."""
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]
    
    def world2voxel(self, world):
        """Convert world coordinates to voxel coordinates."""
        return (world - self.pc_range[:3][None, :]) / self.voxel_size[None, :]