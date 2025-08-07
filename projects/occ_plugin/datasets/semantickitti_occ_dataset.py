import os
import pickle
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import SemanticKITTIDataset
from projects.occ_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results


@DATASETS.register_module()
class SemanticKittiOCCDataset(SemanticKITTIDataset):
    """Occupancy dataset for SemanticKITTI.

    This dataset converts the SemanticKITTI annotation format to the format
    expected by our occupancy pipelines. Camera calibration information is not
    provided in the public SemanticKITTI labels, therefore identity matrices are
    used as placeholders.
    """

    def __init__(self, occ_size, pc_range, occ_root, **kwargs):
        super().__init__(**kwargs)
        self.ann_file = kwargs["ann_file"]
        self.data_path = kwargs["data_root"]
        data = pickle.load(open(self.ann_file, "rb"))
        self.data_infos = data['data_list']
        self.metainfo = data.get('metainfo', {})
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root
        self._set_group_flag()

    def __getitem__(self, idx):
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
        return self.pipeline(input_dict)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        return self.pipeline(input_dict)

    def get_data_info(self, index):
        sample = self.data_infos[index]

        lidar_path = sample['lidar_points']['lidar_path']
        seq = lidar_path.split('/')[1]
        frame = os.path.splitext(os.path.basename(lidar_path))[0]

        img_path = os.path.join(self.data_path, 'sequences', seq, 'image_2', f'{frame}.png')
        image_paths = [img_path]

        lidar2img_rts = [np.eye(4, dtype=np.float32)]
        cam_positions = [np.zeros(3, dtype=np.float32)]
        focal_positions = [np.array([0, 0, 1], dtype=np.float32)]

        occ_label_path = os.path.join(self.data_path, sample['pts_semantic_mask_path'])
        pts_filename = os.path.join(self.data_path, lidar_path)

        curr = {
            'cams': {
                'CAM_FRONT': {
                    'data_path': img_path,
                    'cam_intrinsic': np.eye(3, dtype=np.float32),
                    'sensor2lidar_rotation': np.eye(3, dtype=np.float32),
                    'sensor2lidar_translation': np.zeros(3, dtype=np.float32),
                }
            },
            'lidar_path': pts_filename,
            'scene_token': sample['sample_id'],
            'sample_idx': sample['sample_id'],
            'timestamp': 0,
        }

        lidar2cam_dic = {'CAM_FRONT': np.eye(4, dtype=np.float32)}

        input_dict = {
            'pts_filename': pts_filename,
            'img_filename': image_paths,
            'lidar2img': lidar2img_rts,
            'cam_positions': cam_positions,
            'focal_positions': focal_positions,
            'occ_label_path': occ_label_path,
            'sample_id': sample['sample_id'],
            'scene_token': sample['sample_id'],
            'lidar_token': f"{sample['sample_id']}_{index:06d}",
            'frame_idx': index,
            'curr': curr,
            'lidar2cam_dic': lidar2cam_dic,
        }
        return input_dict

    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}

        evaluation_semantic = sum(results['SC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results[f'SC_{key}'] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)

        evaluation_semantic = sum(results['SSC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results[f'SSC_{key}'] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)

        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results[f'SSC_fine_{key}'] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)

        return eval_results
