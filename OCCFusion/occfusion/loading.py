# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional
import copy
import mmcv
import numpy as np
import torch
import os
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles, Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS
from nuscenes.utils.data_classes import RadarPointCloud
import pykitti.utils as utils
import yaml
from scipy.spatial.transform import Rotation

@TRANSFORMS.register_module()
class SemanticKITTI_Image_Load(LoadMultiViewImageFromFiles):
    
    def get_cam_mtx(self,filepath):
        data = np.loadtxt(filepath)
        P = np.zeros((3,3))
        P[0,0] = data[0]
        P[1,1] = data[1]
        P[2,2] = 1
        P[0,2] = data[2]
        P[1,2] = data[3]
        return P
    
    def get_lidar2cam_mtx(self,filepath):
        with open(filepath,'r') as f:
            data = yaml.load(f,Loader= yaml.Loader)
        q = data['os1_cloud_node-pylon_camera_node']['q']
        q = np.array([q['x'],q['y'],q['z'],q['w']])
        t = data['os1_cloud_node-pylon_camera_node']['t']
        t = np.array([t['x'],t['y'],t['z']])
        R_vc = Rotation.from_quat(q)
        R_vc = R_vc.as_matrix()

        RT = np.eye(4,4)
        RT[:3,:3] = R_vc
        RT[:3,-1] = t
        RT = np.linalg.inv(RT)
        return RT
    
    def transform(self, result: dict) -> Optional[dict]:
        #print(result)
        Tr_4x4 = self.get_lidar2cam_mtx(result['transforms_path'])
        K_3x3 = self.get_cam_mtx(result['camera_info_path'])
        P_4x4 = np.eye(4)
        P_4x4[:3, :3] = K_3x3
        calib_mtx = P_4x4 @ Tr_4x4
        result['lidar2img'] = np.stack([calib_mtx], axis=0)
        
        img_byte = get(result['img_path'], backend_args=self.backend_args) 
        img = mmcv.imfrombytes(img_byte, flag=self.color_type)
        result['img'] = [img]
        
        #import pdb; pdb.set_trace()
        img_path = result.get('img_path', '')
        if img_path:
            parts = img_path.split(os.sep)
            # 예를 들어, parts[0]가 sample_id가 되고, 파일명에서 frame_id를 추출
            sample_id = parts[8]  # 예: "00004"
            file_name = os.path.basename(img_path)  # 예: "frame000400.png"
            frame_id = None
            if file_name.startswith("frame"):
                # "frame" 이후 6자리 숫자가 frame_id라고 가정
                frame_id = file_name[5:11]
            result['sample_id'] = sample_id
            result['frame_id'] = frame_id
            


        return result


@TRANSFORMS.register_module()
class LoadSemanticKITTI_Occupancy(BaseTransform):
    def get_remap_lut(self, label_map):
        maxkey = max(label_map.keys())
        # 기본값을 255로 채워서, label_map에 없는 인덱스는 모두 255가 되도록 함
        remap_lut = np.full((maxkey + 1), 255, dtype=np.int32)
        # label_map 예시: {0: 1, 1: 2, 255: 255}
        for key, value in label_map.items():
            remap_lut[key] = value
        return remap_lut

    def unpack(self, compressed):
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1
        return uncompressed

    def transform(self, result: dict) -> dict:
       
        gt = np.fromfile(result['pts_semantic_mask_path'], dtype=np.uint8).astype(np.float32)
        #print(f"Original data size: {gt.shape[0]}")
        #print(f"Expected size: {192 * 256 * 40}")
        #print(f"File path: {result['pts_semantic_mask_path']}")
        
        #print(f"Unique values: {np.unique(gt)}")

        if 'voxel_invalid_path' in result and result['voxel_invalid_path'] is not None:
            invalid = np.fromfile(result['voxel_invalid_path'], dtype=np.uint8)
            invalid = self.unpack(invalid)
            # invalid==1 => 255
            gt[np.isclose(invalid, 1)] = 255

        # (선택) remap lut
        #if 'label_mapping' in result:
            #remap_lut = self.get_remap_lut(result['label_mapping'])
            #gt = remap_lut[gt.astype(np.uint16)].astype(np.float32)

        # reshape
        gt_masked = gt.reshape([256, 256, 32] ) 

        gt_masked = torch.from_numpy(gt_masked)

        idx_masked = torch.where(gt_masked != 255)

        # idx_masked now includes positions with 0,1,2,3
        label_masked = gt_masked[idx_masked[0], idx_masked[1], idx_masked[2]]
        
        # semantickitti_occ_masked = [x, y, z, label]
        semantickitti_occ_masked = torch.stack(
            [idx_masked[0], idx_masked[1], idx_masked[2], label_masked], dim=1
        ).long()

        #print(semantickitti_occ_masked.shape)
        result['occ_semantickitti_masked'] = semantickitti_occ_masked
        #result['occ_trajectory'] = np.load(result['geom_occ_path'].replace("geom","save_0315"))['filtered_mask']
        return result

    def __repr__(self) -> str:
        return self.__class__.__name__


@TRANSFORMS.register_module()
class LoadSemanticKITTI_Lidar(BaseTransform):
    def __init__(self,
                 pc_range=None,
                 with_fov=False,
                 img_width=1920,
                 img_height=1200):
        self.pc_range = pc_range
        self.with_fov = with_fov
        self.img_width = img_width
        self.img_height = img_height

    def project_points_to_image(self, points, lidar2img):
        """
        라이다 좌표계의 포인트를 이미지 평면에 투영
        """
        if len(points) == 0:
            return np.array([])
            
        # homogeneous coordinates로 변환
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
        
        # 라이다 좌표계 -> 이미지 평면으로 변환
        proj = (lidar2img @ points_homo.T).T  # (N, 3)
        
        # normalize
        depths = proj[:, 2]
        mask_valid = depths > 0

        if not mask_valid.any():
            return np.array([])
            
        u = proj[:, 0] / depths
        v = proj[:, 1] / depths
        
        # 이미지 범위 내 점 선택 (mask_valid인 점들 중에서)
        mask_img = mask_valid & (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)
        valid_indices = np.where(mask_img)[0]
        
        return valid_indices

    def transform(self, result: dict) -> dict:
        # 라이다 포인트 로드
        lidar_path = result['lidar_points']['lidar_path']
        pts = utils.load_velo_scan(lidar_path)
        pts = torch.from_numpy(pts)

        # pc_range로 먼저 필터링
        if self.pc_range is not None:
            idx = torch.where(
                (pts[:, 0] > self.pc_range[0]) &
                (pts[:, 1] > self.pc_range[1]) &
                (pts[:, 2] > self.pc_range[2]) &
                (pts[:, 0] < self.pc_range[3]) &
                (pts[:, 1] < self.pc_range[4]) &
                (pts[:, 2] < self.pc_range[5])
            )
            pts = pts[idx[0]]

        # FOV 필터링
        if self.with_fov:
            if 'lidar2img' not in result:
                raise KeyError("lidar2img matrix is required for FOV filtering")
            
            # Numpy로 변환하여 projection 수행
            pts_np = pts.numpy()
            lidar2img = result['lidar2img']
            
            # FOV 내 포인트 인덱스 얻기
            valid_indices = self.project_points_to_image(pts_np[:, :3], lidar2img)
            
            # 유효한 포인트만 선택
            if len(valid_indices) > 0:
                pts = pts[valid_indices]
            else:
                pts = pts.new_zeros((0, pts.shape[1]))

        result['points'] = pts
        return result

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(pc_range={self.pc_range}, '
        repr_str += f'with_fov={self.with_fov}, '
        repr_str += f'img_width={self.img_width}, '
        repr_str += f'img_height={self.img_height})'
        return repr_str

@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename, cam2img, lidar2cam, lidar2img, ego2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])
            
            ego2cam_array = np.linalg.inv(np.array(cam_item['cam2ego'],dtype=np.float64))
            lidar2cam_array = np.array(cam_item['lidar2cam'],dtype=np.float64)
            cam2img_array = np.eye(4).astype(np.float64)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img'],dtype=np.float64)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)
            ego2img.append(cam2img_array @ ego2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)
        results['ego2img'] = np.stack(ego2img, axis=0)
        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        # gbr follow tpvformer
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # imgs = [
        #     cv2.resize(mmcv.imfrombytes(img_byte, flag=self.color_type,backend='cv2'),(0,0),fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
        #     for img_byte in img_bytes
        # ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 0.2 # 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results


@TRANSFORMS.register_module()
class SegLabelMapping(BaseTransform):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def transform(self, results: dict) -> dict:
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        assert 'seg_label_mapping' in results
        label_mapping = results['seg_label_mapping']
        converted_pts_sem_mask = np.vectorize(
            label_mapping.__getitem__, otypes=[np.uint8])(
                pts_semantic_mask)

        results['pts_semantic_mask'] = converted_pts_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_pts_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadOccupancy(BaseTransform):
    def __init__(self,
                use_occ3d=False,
                pc_range=None):
       self.use_occ3d = use_occ3d
       self.pc_range = pc_range
       
    def transform(self, results: dict) -> dict:
        
        if self.use_occ3d:
            lidar2ego_rotation = np.array(results['lidar_points']['lidar2ego'])[:3,:3]
            lidar2ego_translation = np.array(results['lidar_points']['lidar2ego'])[:3,-1]
            points = results['points'].numpy()
            points[:,:3] = points[:,:3] @ lidar2ego_rotation.T
            points[:,:3] += lidar2ego_translation
            points = torch.from_numpy(points)
            idx = torch.where((points[:,0] > self.pc_range[0])
                            & (points[:,1] > self.pc_range[1])
                            & (points[:,2] > self.pc_range[2])
                            & (points[:,0] < self.pc_range[3])
                            & (points[:,1] < self.pc_range[4])
                            & (points[:,2] < self.pc_range[5]))
            points = points[idx]
            results['points'] = points
            
            occ_3d_folder = results['lidar_points']['lidar_path'].split('samples')[0] + 'Occ3D'
            occ_3d_path = os.path.join(occ_3d_folder, results['token'], 'labels.npz')
            occ_3d = np.load(occ_3d_path)
            occ_3d_semantic = occ_3d['semantics']
            occ_3d_cam_mask = occ_3d['mask_camera']
            occ_3d_semantic[occ_3d_semantic==0]=18 # now class 1~18, with 18:others
            occ_3d_gt_masked = occ_3d_semantic * occ_3d_cam_mask
            occ_3d_gt_masked[occ_3d_gt_masked==0]=255 # invisible voxels
            occ_3d_gt_masked[occ_3d_gt_masked==17]=0
            occ_3d_gt_masked[occ_3d_gt_masked==18]=17
            
            occ_3d_gt_masked = torch.from_numpy(occ_3d_gt_masked)
            idx_masked = torch.where(occ_3d_gt_masked > 0)
            label_masked = occ_3d_gt_masked[idx_masked[0],idx_masked[1],idx_masked[2]]
            occ_3d_masked = torch.stack([idx_masked[0],idx_masked[1],idx_masked[2],label_masked],dim=1).long()
            
            occ_3d_semantic[occ_3d_semantic==17]=0
            occ_3d_semantic[occ_3d_semantic==18]=17
            occ_3d_gt = torch.from_numpy(occ_3d_semantic)
            idx = torch.where(occ_3d_gt > 0)
            label = occ_3d_gt[idx[0],idx[1],idx[2]]
            occ3d = torch.stack([idx[0],idx[1],idx[2],label],dim=1).long()
            
            results['occ_3d_masked'] = occ_3d_masked
            results['occ_3d'] = occ3d
            
        else:
            occ_file_name = results['lidar_points']['lidar_path'].split('/')[-1] + '.npy'
            occ_200_folder = results['lidar_points']['lidar_path'].split('samples')[0] + 'occ_samples'
            occ_200_path = os.path.join(occ_200_folder, occ_file_name)
            occ_200 = np.load(occ_200_path)
            occ_200[:,3][occ_200[:,3]==0]=255
            # occ_200[:,3] = 1 # for IoU Task
            occ_200 = torch.from_numpy(occ_200)
            results['occ_200'] = occ_200
            
        return results
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class LoadRadarPointsMultiSweeps(BaseTransform):
   """Load radar points from multiple sweeps.
   This is usually used for nuScenes dataset to utilize previous sweeps.
   Args:
       sweeps_num (int): Number of sweeps. Defaults to 10.
       load_dim (int): Dimension number of the loaded points. Defaults to 5.
       use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
   """


   def __init__(self,
                use_occ3d=False,
                load_dim=18,
                use_dim=[0, 1, 2, 8, 9, 18],
                sweeps_num=5,
                pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]):
       self.use_occ3d = use_occ3d
       self.load_dim = load_dim
       self.use_dim = use_dim
       self.sweeps_num = sweeps_num
       self.pc_range = pc_range


   def _load_points(self, pts_filename):
       """Private function to load point clouds data.
       Args:
           pts_filename (str): Filename of point clouds data.
       Returns:
           np.ndarray: An array containing point clouds data.
           [N, 18]
       """
       radar_obj = RadarPointCloud.from_file(pts_filename)


       #[18, N]
       points = radar_obj.points


       return points.transpose().astype(np.float32)


   def __call__(self, results):
       """Call function to load multi-sweep point clouds from files.
       Args:
           results (dict): Result dict containing multi-sweep point cloud \
               filenames.
       Returns:
           dict: The result dict containing the multi-sweep points data. \
               Added key and value are described below.
               - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                   cloud arrays.
       """
       radars_dict = results['radars']
       lidar2ego_rotation = np.array(results['lidar_points']['lidar2ego'])[:3,:3]
       lidar2ego_translation = np.array(results['lidar_points']['lidar2ego'])[:3,-1]
       points_sweep_list = []
       for key, sweeps in radars_dict.items():
           if len(sweeps) < self.sweeps_num:
               idxes = list(range(len(sweeps)))
           else:
               idxes = list(range(self.sweeps_num))
          
           ts = sweeps[0]['timestamp'] * 1e-6
           for idx in idxes:
               sweep = sweeps[idx]


               points_sweep = self._load_points(sweep['data_path'])
               points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)


               timestamp = sweep['timestamp'] * 1e-6
               time_diff = ts - timestamp
               time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff


               # velocity compensated by the ego motion in sensor frame
               velo_comp = points_sweep[:, 8:10]
               velo_comp = np.concatenate((velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
               velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
               if self.use_occ3d:
                   velo_comp = velo_comp @ lidar2ego_rotation.T
               velo_comp = velo_comp[:, :2]


               # velocity in sensor frame
               velo = points_sweep[:, 6:8]
               velo = np.concatenate((velo, np.zeros((velo.shape[0], 1))), 1)
               velo = velo @ sweep['sensor2lidar_rotation'].T
               if self.use_occ3d:
                   velo = velo @ lidar2ego_rotation.T
               velo = velo[:, :2]


               points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
               points_sweep[:, :3] += sweep['sensor2lidar_translation']
               if self.use_occ3d:
                   points_sweep[:, :3] = points_sweep[:, :3] @ lidar2ego_rotation.T
                   points_sweep[:, :3] += lidar2ego_translation

               points_sweep_ = np.concatenate(
                   [points_sweep[:, :6], velo,
                    velo_comp, points_sweep[:, 10:],
                    time_diff], axis=1)
               points_sweep_list.append(points_sweep_)
      
       points = np.concatenate(points_sweep_list, axis=0)
      
       points = points[:, self.use_dim]
      
       points = torch.from_numpy(points)
      
       results['radars'] = points
       return self.transform(results)


   def transform(self, results):
       radar_pts = results['radars']
       radar_pts_xyz = radar_pts[:,0:3]
       idx = torch.where((radar_pts_xyz[:,0] > self.pc_range[0])
                         & (radar_pts_xyz[:,1] > self.pc_range[1])
                         & (radar_pts_xyz[:,2] > self.pc_range[2])
                         & (radar_pts_xyz[:,0] < self.pc_range[3])
                         & (radar_pts_xyz[:,1] < self.pc_range[4])
                         & (radar_pts_xyz[:,2] < self.pc_range[5]))
       radar_pts = radar_pts[idx]
       results['radars'] = radar_pts
       return results
  
   def __repr__(self):
       """str: Return a string that describes the module."""
       return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'
