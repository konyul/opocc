import os
import numpy as np
import pickle
from mmdet.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
import mmcv
from os import path as osp
import torch
from mmcv.parallel import DataContainer as DC
import random


@DATASETS.register_module()
class Rellis3DDataset(Custom3DDataset):
    """Rellis-3D Dataset for 3D occupancy prediction.
    
    This dataset loader is designed to work with OCCFusion-style data structure
    for Rellis-3D dataset with 3 semantic classes.
    """
    
    CLASSES = ('empty', 'traverse', 'non_traverse')
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 occ_size=None,
                 pc_range=None,
                 occ_root=None,
                 test_mode=False,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 queue_length=1,
                 **kwargs):
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root if occ_root is not None else data_root
        self.queue_length = queue_length
        
        # Remove use_valid_flag from kwargs if present
        kwargs.pop('use_valid_flag', None)
        
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            test_mode=test_mode,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            **kwargs
        )
        
    def load_annotations(self, ann_file):
        """Load annotations from pickle file."""
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def get_data_info(self, index):
        """Get data info according to the given index.
        
        Args:
            index (int): Index of the sample data to get.
            
        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines.
        """
        info = self.data_infos[index]
        
        # Basic sample info
        input_dict = dict(
            sample_idx=info.get('token', f'sample_{index}'),
            pts_filename=info.get('lidar_path', ''),
            timestamp=info.get('timestamp', 0)
        )
        
        # Add image paths - Rellis3D has single front camera
        if 'cams' in info:
            img_filenames = []
            lidar2img_rts = []
            cam_intrinsics = []
            
            # For Rellis3D, we only have front camera
            cam_info = info['cams'].get('CAM_FRONT', {})
            if cam_info:
                img_filenames.append(cam_info['data_path'])
                
                # Camera intrinsics
                intrinsic = cam_info.get('cam_intrinsic', np.eye(3))
                cam_intrinsics.append(intrinsic)
                
                # Lidar to camera transformation
                lidar2cam_r = cam_info.get('sensor2lidar_rotation', np.eye(3))
                lidar2cam_t = cam_info.get('sensor2lidar_translation', np.zeros(3))
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[:3, 3] = -lidar2cam_r.T @ lidar2cam_t
                
                # Camera to image transformation
                lidar2img_rt = intrinsic @ lidar2cam_rt[:3, :]
                lidar2img_rts.append(lidar2img_rt)
            
            input_dict.update(dict(
                img_filename=img_filenames,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics
            ))
        
        # Add 3D bounding boxes if available
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            
        # Add occupancy GT path
        if self.occ_root is not None:
            occ_gt_path = osp.join(self.occ_root, f'{input_dict["sample_idx"]}_occupancy.npy')
            input_dict['occ_gt_path'] = occ_gt_path
            
        return input_dict
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.
        
        Args:
            index (int): Index of the annotation data to get.
            
        Returns:
            dict: Annotation information.
        """
        info = self.data_infos[index]
        
        # For occupancy prediction, we mainly need GT occupancy labels
        # 3D bounding boxes are optional
        ann_info = dict()
        
        # Get 3D bounding boxes if available
        if 'gt_boxes' in info:
            gt_bboxes_3d = info['gt_boxes']
            gt_labels_3d = info['gt_names']
            
            # Convert to numpy arrays
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
            gt_labels_3d = np.array([self.CLASSES.index(name) if name in self.CLASSES else -1 
                                   for name in gt_labels_3d], dtype=np.int64)
            
            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
            ann_info['gt_labels_3d'] = gt_labels_3d
        else:
            # Empty annotations
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0,), dtype=np.int64)
            
        return ann_info
    
    def prepare_train_data(self, index):
        """Training data preparation.
        
        Args:
            index (int): Index for accessing the target data.
            
        Returns:
            dict: Training data dict of the corresponding index.
        """
        if self.queue_length > 1:
            # Multi-frame training
            queue = []
            index_list = list(range(index-self.queue_length+1, index+1))
            for i in index_list:
                i = max(0, min(i, len(self) - 1))
                input_dict = self.get_data_info(i)
                if input_dict is None:
                    return None
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                queue.append(example)
            return self.union2one(queue)
        else:
            # Single frame training
            input_dict = self.get_data_info(index)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            return example
    
    def prepare_test_data(self, index):
        """Test data preparation.
        
        Args:
            index (int): Index for accessing the target data.
            
        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def union2one(self, queue):
        """Convert multi-frame queue to single sample."""
        imgs_list = [each['img'].data for each in queue]
        points_list = [each.get('points', None) for each in queue]
        
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        if points_list[0] is not None:
            queue[-1]['points'] = DC(torch.stack([p.data for p in points_list]), 
                                   cpu_only=False, stack=True)
        
        # Keep only the last frame's metadata
        queue = queue[-1]
        return queue
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='occ',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in occupancy prediction protocol.
        
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            result_name (str): Result name in the metric prefix.
                Default: 'occ'.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
                
        Returns:
            dict[str, float]: Evaluation results.
        """
        # For occupancy evaluation, we'll implement IoU and mIoU metrics
        eval_results = {}
        
        if 'occ' in metric or 'occupancy' in metric:
            # Collect predictions and ground truths
            all_pred_occ = []
            all_gt_occ = []
            
            for i, result in enumerate(results):
                # Get prediction
                pred_occ = result.get('occ_pred', None)
                if pred_occ is not None:
                    all_pred_occ.append(pred_occ)
                    
                    # Get ground truth
                    info = self.data_infos[i]
                    occ_gt_path = osp.join(self.occ_root, f'{info["token"]}_occupancy.npy')
                    if osp.exists(occ_gt_path):
                        gt_occ = np.load(occ_gt_path)
                        all_gt_occ.append(gt_occ)
            
            # Calculate metrics
            if len(all_pred_occ) > 0 and len(all_gt_occ) > 0:
                all_pred_occ = np.stack(all_pred_occ)
                all_gt_occ = np.stack(all_gt_occ)
                
                # Calculate IoU for each class
                num_classes = 3  # empty, traversable, non-traversable
                ious = []
                
                for cls_idx in range(num_classes):
                    pred_mask = all_pred_occ == cls_idx
                    gt_mask = all_gt_occ == cls_idx
                    
                    intersection = np.logical_and(pred_mask, gt_mask).sum()
                    union = np.logical_or(pred_mask, gt_mask).sum()
                    
                    if union > 0:
                        iou = intersection / union
                    else:
                        iou = 1.0 if intersection == 0 else 0.0
                    
                    ious.append(iou)
                    eval_results[f'{result_name}_IoU_class_{cls_idx}'] = iou
                
                # Calculate mean IoU
                miou = np.mean(ious)
                eval_results[f'{result_name}_mIoU'] = miou
                
                # Calculate overall accuracy
                accuracy = (all_pred_occ == all_gt_occ).mean()
                eval_results[f'{result_name}_accuracy'] = accuracy
                
        return eval_results


@DATASETS.register_module()
class CustomNuScenesOccDataset(Rellis3DDataset):
    """Wrapper to make Rellis3D dataset compatible with NuScenes-based pipelines."""
    
    def __init__(self, **kwargs):
        # Remove any unexpected arguments that might be passed
        kwargs.pop('use_valid_flag', None)
        kwargs.pop('use_camera', None)
        kwargs.pop('use_lidar', None)
        kwargs.pop('use_radar', None)
        kwargs.pop('use_map', None)
        kwargs.pop('use_external', None)
        super().__init__(**kwargs)