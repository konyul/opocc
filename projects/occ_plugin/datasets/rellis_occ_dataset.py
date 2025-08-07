import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import SemanticKITTIDataset
import os
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.occ_plugin.utils.formating import (
    cm_to_ious, format_SC_results, format_SSC_results_rellis)
import pickle
@DATASETS.register_module()
class rellisOCCDataset(SemanticKITTIDataset):
    def __init__(self, occ_size, pc_range, occ_root, **kwargs):
        super().__init__(**kwargs)
        self.ann_file = kwargs["ann_file"]
        self.data_path = kwargs["data_root"]
        self.data_infos = pickle.load(open(self.ann_file,"rb"))['data_list']
        self.metainfo = pickle.load(open(self.ann_file,"rb")).get('metainfo', {})
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root
        self._set_group_flag()

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
            
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            return data

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):

        sample = self.data_infos[index]
        
        """Convert OCCFusion format to GaussianFormer format"""
        # Load camera calibration
        calib_path = os.path.join(self.data_path, sample['calib_txt_path'])
        camera_info_path = os.path.join(self.data_path, sample['camera_info_path'])
        transforms_path = os.path.join(self.data_path, sample['transforms_path'])
        
        # Rellis-3D has single camera image
        img_path = os.path.join(self.data_path, sample['img_path'])
        
        # Single camera setup
        image_paths = [img_path]
        
        # Load calibration data
        lidar2img_rts = []
        cam_positions = []
        focal_positions = []
        
        # Load actual calibration for Rellis-3D if available
        # For now, create transforms that duplicate the single camera view
        for i in range(1):
            # TODO: Load actual calibration from calib_txt_path
            # For now, use identity transform
            lidar2img = np.eye(4, dtype=np.float32)
            lidar2img_rts.append(lidar2img)
            cam_positions.append(np.array([0, 0, 0], dtype=np.float32))
            focal_positions.append(np.array([0, 0, 1], dtype=np.float32))
        
        # Load occupancy labels
        # Check if pts_semantic_mask_path is already full path
        if sample['pts_semantic_mask_path'].startswith('/'):
            occ_label_path = sample['pts_semantic_mask_path']
        else:
            occ_label_path = os.path.join(self.data_path, sample['pts_semantic_mask_path'])
        
        # Add point cloud filename
        if 'lidar_points' in sample and 'lidar_path' in sample['lidar_points']:
            pts_filename = os.path.join(self.data_path, sample['lidar_points']['lidar_path'])
        else:
            pts_filename = ''
        
        # Create curr structure for BEVDet pipeline
        curr = {
            'cams': {
                'CAM_FRONT': {
                    'data_path': img_path,
                    'cam_intrinsic': np.eye(3, dtype=np.float32),  # TODO: Load actual intrinsics
                    'sensor2lidar_rotation': np.eye(3, dtype=np.float32),
                    'sensor2lidar_translation': np.zeros(3, dtype=np.float32),
                }
            },
            'lidar_path': pts_filename,
            'scene_token': sample['sample_id'],
            'sample_idx': sample['sample_id'],
            'timestamp': 0,
        }
        
        # Create lidar2cam dictionary
        lidar2cam_dic = {
            'CAM_FRONT': np.eye(4, dtype=np.float32)  # TODO: Load actual calibration
        }
        
        input_dict = {
            'pts_filename': pts_filename,  # Required by LoadPointsFromFile
            'img_filename': image_paths,
            'lidar2img': lidar2img_rts,
            'cam_positions': cam_positions,
            'focal_positions': focal_positions,
            'occ_label_path': occ_label_path,
            'sample_id': sample['sample_id'],
            'scene_token': sample['sample_id'],  # Use sample_id as scene token
            'lidar_token': f"{sample['sample_id']}_{index:06d}",  # Create unique lidar token
            'frame_idx': index,
            'curr': curr,  # BEVDet format
            'lidar2cam_dic': lidar2cam_dic,  # BEVDet format
        }

        return input_dict


    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        ''' evaluate SC '''
        evaluation_semantic = sum(results['SC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results['SC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC '''
        evaluation_semantic = sum(results['SSC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results_rellis(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['SSC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results_rellis(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)
            
        return eval_results
