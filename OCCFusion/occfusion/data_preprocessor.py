# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
import matplotlib.pyplot as plt
from mmdet3d.models import Det3DDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class OccFusionDataPreprocessor(Det3DDataPreprocessor):
    
    def visualize_raw_pointcloud_bev(self, points_list, save_path=None):
        """
        원본 Point cloud를 bird's eye view로 시각화합니다.
        
        Args:
            points_list (List[Tensor]): List of raw point clouds
            save_path (str, optional): 이미지를 저장할 경로
        """
        plt.figure(figsize=(15, 5 * len(points_list)))
        
        for batch_idx, points in enumerate(points_list):
            # CUDA 텐서를 CPU로 이동 후 넘파이로 변환
            points = points.cpu().numpy()
            
            plt.subplot(len(points_list), 1, batch_idx + 1)
            
            # x, y 좌표 추출
            x = points[:, 0]  # x coordinates
            y = points[:, 1]  # y coordinates
            
            # scatter plot 생성
            scatter = plt.scatter(x, y, s=1, c=points[:, 2], cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, label='Z height')
            
            # plot 설정
            plt.title(f'Raw Point Cloud BEV - Batch {batch_idx}')
            plt.xlabel('X axis (meters)')
            plt.ylabel('Y axis (meters)')
            plt.grid(True)
            
            # 축 범위 자동 설정을 위해 데이터의 min/max 사용
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            # 여백 추가
            margin = 2
            plt.xlim([x_min - margin, x_max + margin])
            plt.ylim([y_min - margin, y_max + margin])

        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'raw_{save_path}')
            print(f"Saved raw point cloud visualization to raw_{save_path}")
        else:
            plt.show()
        
        plt.close()

    
    
    
    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)
        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            #self.visualize_raw_pointcloud_bev(inputs['points'], save_path="pointcloud_bev_fov.png")
            #import pdb; pdb.set_trace()
            batch_inputs['points'] = inputs['points']
                
            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict
            
            # Create sparse voxel feature based on the VoxelNet for lidar points
            batch_lidar_voxel_feats = []
            batch_lidar_voxel_coord = []
            for point, data_sample in zip(inputs['points'], data_samples):
                lidar_voxel_coords = data_sample.point_coors
                lidar_voxel_coords = lidar_voxel_coords[:,[2,0,1]] # put D to first dimension
                lidar_voxel_coords, inv_ind, voxel_counts = torch.unique(lidar_voxel_coords, dim=0, return_inverse=True, return_counts=True)
                batch_lidar_voxel_coord.append(lidar_voxel_coords.long())
                lidar_voxel_features = []
                for i in range(len(lidar_voxel_coords)):
                    if point.shape[-1] == 5:
                        voxel=torch.zeros((35,8))
                    else:
                        voxel=torch.zeros((35,7))
                    pts = point[inv_ind == i]
                    if voxel_counts[i] > 35:
                        pts = pts[:35, :]
                        voxel_counts[i] = 35
                    
                    # augment the points
                    voxel[:pts.shape[0], :] = torch.cat((pts, pts[:, :3] - torch.mean(pts[:, :3], 0)), dim=1)
                    lidar_voxel_features.append(voxel)
                batch_lidar_voxel_feats.append(torch.stack(lidar_voxel_features,dim=0).to(inputs['imgs'].device))
            
            batch_inputs['lidar_voxel_feats'] = batch_lidar_voxel_feats
            batch_inputs['lidar_voxel_coords'] = batch_lidar_voxel_coord
        
        if 'radars' in inputs:
            batch_radar_voxel_feats = []
            batch_radar_voxel_coord = []
            for radar_pts in inputs['radars']:
                min_bound = radar_pts.new_tensor(self.voxel_layer.point_cloud_range[:3])
                radar_voxel_coors = torch.floor((radar_pts[:,:3] - min_bound) / radar_pts.new_tensor(self.voxel_layer.voxel_size)).int()
                radar_voxel_coors = radar_voxel_coors[:,[2,0,1]] # put D to first dimension
                radar_voxel_coors, inv_ind, voxel_counts = torch.unique(radar_voxel_coors, dim=0, return_inverse=True, return_counts=True)
                batch_radar_voxel_coord.append(radar_voxel_coors.long())
                radar_voxel_features = []
                for i in range(len(radar_voxel_coors)):
                    voxel=torch.zeros((35,11))
                    pts = radar_pts[inv_ind == i]
                    if voxel_counts[i] > 35:
                        pts = pts[:35, :]
                        voxel_counts[i] = 35
                    
                    # augment the points
                    voxel[:pts.shape[0], :] = torch.cat((pts, pts[:, :5] - torch.mean(pts[:, :5], 0)), dim=1)
                    radar_voxel_features.append(voxel)
                batch_radar_voxel_feats.append(torch.stack(radar_voxel_features,dim=0).to(inputs['imgs'].device))
            
            batch_inputs['radar_voxel_feats'] = batch_radar_voxel_feats
            batch_inputs['radar_voxel_coords'] = batch_radar_voxel_coord
            
        
        if 'occ_200' in data['inputs']:
            batch_inputs['dense_occ_200'] = data['inputs']['occ_200']

        if 'occ_3d' in data['inputs']:
            batch_inputs['occ_3d'] = data['inputs']['occ_3d']
            
        if 'occ_3d_masked' in data['inputs']:
            batch_inputs['occ_3d_masked'] = data['inputs']['occ_3d_masked']
        
        if 'occ_semantickitti' in data['inputs']:
            batch_inputs['occ_semantickitti'] = data['inputs']['occ_semantickitti']
        
        if 'occ_semantickitti_masked' in data['inputs']:
            batch_inputs['occ_semantickitti_masked'] = data['inputs']['occ_semantickitti_masked']

        if 'occ_trajectory' in data['inputs']:
            batch_inputs['occ_trajectory'] = data['inputs']['occ_trajectory']
        
        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}
    
    @torch.no_grad()
    def voxelize(self, points, data_samples) -> List[Tensor]:
        """Apply voxelization to point cloud. In TPVFormer, it will get voxel-
        wise segmentation label and voxel/point coordinates.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (List[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            List[Tensor]: Coordinates of voxels, shape is Nx3,
        """
        for point, data_sample in zip(points, data_samples):
            min_bound = point.new_tensor(self.voxel_layer.point_cloud_range[:3])
            coors = torch.floor((point[:,:3] - min_bound) / point.new_tensor(self.voxel_layer.voxel_size)).int()
            data_sample.point_coors = coors
