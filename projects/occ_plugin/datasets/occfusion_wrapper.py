import os
import numpy as np
import torch
from PIL import Image
from mmdet.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet.datasets.pipelines import Compose
import mmengine


@DATASETS.register_module()
class OCCFusionWrapper(Custom3DDataset):
    """Wrapper for OCCFusion dataset to work with OpenOccupancy.
    
    This dataset directly loads from OCCFusion pickle files and
    handles the data in a format compatible with OpenOccupancy.
    """
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 occ_size=[256, 256, 32],
                 pc_range=[-25.6, -12.8, -1.6, 0, 12.8, 1.6],
                 test_mode=False,
                 box_type_3d='LiDAR',
                 **kwargs):
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.data_root = data_root
        
        # Remove unexpected arguments
        kwargs.pop('occ_root', None)
        kwargs.pop('use_valid_flag', None)
        
        # Don't call parent init yet - load data first
        self.test_mode = test_mode
        self.box_type_3d = box_type_3d
        self.CLASSES = classes if classes is not None else ['empty', 'traverse', 'non_traverse']
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.modality = modality
        
        # Load annotation file
        self.load_annotations(ann_file)
        
        # Set flag
        self._set_group_flag()
        
    def load_annotations(self, ann_file):
        """Load annotations from OCCFusion pickle file."""
        data = mmengine.load(ann_file)
        self.data_infos = data['data_list']
        self.metainfo = data.get('metainfo', {})
        
    def get_data_info(self, index):
        """Get data info for given index."""
        sample = self.data_infos[index]
        
        # Image path
        img_path = os.path.join(self.data_root, sample['img_path'])
        
        # LiDAR path
        if 'lidar_points' in sample and 'lidar_path' in sample['lidar_points']:
            pts_filename = os.path.join(self.data_root, sample['lidar_points']['lidar_path'])
        else:
            pts_filename = ''
            
        # Occupancy label path
        if sample['pts_semantic_mask_path'].startswith('/'):
            occ_label_path = sample['pts_semantic_mask_path']
        else:
            occ_label_path = os.path.join(self.data_root, sample['pts_semantic_mask_path'])
        
        # Create input dict
        input_dict = {
            'pts_filename': pts_filename,
            'img_filename': [img_path],  # List for compatibility
            'occ_label_path': occ_label_path,
            'sample_idx': sample['sample_id'],
            'scene_token': sample['sample_id'],
            'lidar_token': f"{sample['sample_id']}_{index:06d}",
        }
        
        # Add annotation info if available
        if not self.test_mode:
            # Empty annotations for occupancy task
            input_dict['ann_info'] = {
                'gt_bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                'gt_labels_3d': np.zeros((0,), dtype=np.int64),
            }
            
        return input_dict
    
    def pre_pipeline(self, results):
        """Prepare data dict for pipeline."""
        results['img_prefix'] = None
        results['seg_prefix'] = None
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        
    def prepare_train_data(self, index):
        """Training data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def prepare_test_data(self, index):
        """Test data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def _set_group_flag(self):
        """Set flag."""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        
    def evaluate(self, results, **kwargs):
        """Placeholder for evaluation."""
        return {'placeholder': 0.0}