# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

from mmengine.dataset import BaseDataset

from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class SemanticKittiSegDataset(BaseDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Store `True` when building test or val dataset.
    """
    METAINFO = {
        'classes': (  # 실제 사용하려는 클래스로 수정
            'IoU (Occupied)',
            'Traverse',
            'Non-traverse'
        ),
        'ignore_index': 255,
        'label_mapping': dict([(0,0),(1,1), (2,2)])  # 필요한 매핑으로 수정
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs) -> None:
        metainfo = dict(label2cat={
            i: cat_name
            for i, cat_name in enumerate(self.METAINFO['classes'])
        })
        #import pdb;pdb.set_trace()
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        
        data_list = []
        info['img_path'] = osp.join(self.data_prefix['img_path'], info['img_path'])
        info['pts_semantic_mask_path'] = osp.join(self.data_prefix['pts_semantic_mask_path'],info['pts_semantic_mask_path'])
        info['lidar_points']['lidar_path'] = osp.join(self.data_prefix['lidar'], info['lidar_points']['lidar_path'])
        #info['voxel_gt_path'] = osp.join(self.data_prefix['img_path'],info['voxel_gt_path'])
        #info['voxel_invalid_path'] = osp.join(self.data_prefix['img_path'],info['voxel_invalid_path'])
        info['transforms_path'] = osp.join(self.data_prefix['img_path'],info['transforms_path'])
        info['calib_txt_path'] = osp.join(self.data_prefix['img_path'],info['calib_txt_path'])
        info['camera_info_path'] = osp.join(self.data_prefix['img_path'],info['camera_info_path'])
        if 'geom_occ_path' in info:
            info['geom_occ_path'] = osp.join(self.data_prefix['img_path'],info['geom_occ_path'])
        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        #info['label_mapping'] = self.metainfo['label_mapping']

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode:
            info['eval_ann_info'] = dict()

        data_list.append(info)
        return data_list
