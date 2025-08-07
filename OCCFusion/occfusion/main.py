from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from .custom_focal_loss import FocalLoss
from mmdet3d.models.losses import LovaszLoss
from .loss import geo_scal_loss, sem_scal_loss, elbo_loss, lovasz_softmax
import cv2
from torch.nn import MSELoss
import random
import time
from tqdm import tqdm
import local_aggregate_prob
from torch import Tensor
from sklearn.cluster import KMeans
@MODELS.register_module()
class OccFusion(Base3DSegmentor):
    def __init__(self,
                 use_occ3d,
                 use_lidar,
                 use_radar,
                 data_preprocessor,
                 backbone,
                 neck,
                 view_transformer,
                 svfe_lidar,
                 svfe_radar,
                 occ_head,
                 use_uncertainty=False,
                 use_geometry=False,
                 use_gaussian=False,
                 npy_save_folder=None):
        super().__init__(data_preprocessor=data_preprocessor)

        self.npy_save_folder = npy_save_folder
        if self.npy_save_folder and not os.path.exists(self.npy_save_folder):
            Path(self.npy_save_folder).mkdir(parents=True, exist_ok=True)
        self.occ3d = use_occ3d
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.use_uncertainty = use_uncertainty
        self.img_backbone = MODELS.build(backbone)
        self.img_neck = MODELS.build(neck)
        if self.use_lidar:
            self.svfe_lidar = MODELS.build(svfe_lidar)
        if self.use_radar:
            self.svfe_radar = MODELS.build(svfe_radar)
        self.view_transformer = MODELS.build(view_transformer)
        self.occ_head = MODELS.build(occ_head)
        self.loss_fl = FocalLoss(gamma=2,ignore_index=255) # 0: noise label weights=
        self.loss_lovasz = LovaszLoss(loss_type='multi_class',
                                      per_sample=False,
                                      reduction='none')
        self.use_geometry = use_geometry
        if self.use_geometry == 'MSE':
            self.geometric_loss = MSELoss()
            self.geometric_layer = nn.Sequential(
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3)
                )
        self._log_tensor_cache = {}
        self._eps = 1e-8
        self.use_gaussian = use_gaussian
        if self.use_gaussian:
            self.gaussian_decoder = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 8)  # scale(3), rotation(4), opacity(1), logits(3), geometry(3)
            )
            cuda_kwargs = dict(scale_multiplier=1, H=256, W=256, D=32, pc_min=[-25.6, -12.8, -1.6], grid_size=0.1)
            self.aggregator = local_aggregate_prob.LocalAggregator(**cuda_kwargs)

    def multiscale_supervision(self, gt_occ, ratio, gt_shape):
        # Convert numpy array to tuple if needed
        if isinstance(gt_shape, np.ndarray):
            gt_shape = tuple(gt_shape.tolist())
        
        gt = torch.full(
            gt_shape,
            255,
            dtype=torch.long,
            device=gt_occ[0].device
        )
        
        # Vectorized computation for faster processing
        ratio_tensor = torch.tensor(ratio, device=gt_occ[0].device, dtype=torch.float32)
        
        for i in range(gt.shape[0]):
            if len(gt_occ[i]) == 0:
                continue
                
            # Vectorized coordinate computation - much faster than individual operations
            coords_float = gt_occ[i][:, :3].float()
            coords_scaled = torch.div(coords_float, ratio_tensor, rounding_mode='floor').long()
            
            # Bounds checking to prevent index errors
            valid_mask = (
                (coords_scaled[:, 0] >= 0) & (coords_scaled[:, 0] < gt_shape[1]) &
                (coords_scaled[:, 1] >= 0) & (coords_scaled[:, 1] < gt_shape[2]) &
                (coords_scaled[:, 2] >= 0) & (coords_scaled[:, 2] < gt_shape[3])
            )
            
            if valid_mask.any():
                valid_coords = coords_scaled[valid_mask]
                valid_labels = gt_occ[i][valid_mask, 3]
                gt[i, valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = valid_labels

        return gt
    
    def visualize_point_cloud(self, points, save_path):
        """
        Visualize point cloud as a 2D top-down view and save as PNG.
        
        Args:
            points (torch.Tensor): Point cloud data, typically with shape [N, 3+C] 
                                where first 3 dimensions are xyz coordinates
            save_path (str): Path to save the visualization.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
        
        # First check if points is a list (batch of tensors)
        if isinstance(points, list):
            # If it's a list of tensors, concatenate them after moving to CPU
            points_list = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.array(p) for p in points]
            points_np = np.concatenate(points_list, axis=0)
        elif isinstance(points, torch.Tensor):
            # Make sure to move to CPU before converting to numpy
            points_np = points.detach().cpu().numpy()
        else:
            points_np = np.array(points)
        
        # Extract x, y, z coordinates
        x = points_np[:, 0]
        y = points_np[:, 1]
        z = points_np[:, 2]
        
        # Create figure with grid
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # If there are too many points, sample them for better visualization
        max_points = 100000  # Increased for better density
        if len(x) > max_points:
            indices = np.random.choice(len(x), max_points, replace=False)
            x = x[indices]
            y = y[indices]
            z = z[indices]
        
        # Create scatter plot with Z height as color
        scatter = ax.scatter(x, y, c=z, s=0.5, cmap='viridis', alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Z height')
        
        # Set labels and title
        ax.set_xlabel('X axis (meters)')
        ax.set_ylabel('Y axis (meters)')
        ax.set_title(f'Raw Point Cloud BEV - Batch 0')
        
        # Add grid
        ax.grid(True)
        
        # Adjust axis limits to match the image style
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f'Point cloud BEV visualization saved to {save_path} ({len(x)} points)')
            
    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        
        return img_feats_reshaped    

    def normalize(self, array): 
        tensor = torch.tensor(array, dtype=torch.float32)
        mask = tensor != 255
        valid = tensor[mask]
        if len(valid) == 0:
            return tensor
        min_val = valid.min()
        max_val = valid.max()
        range_val = max_val - min_val
        if range_val < self._eps:
            normalized = torch.zeros_like(tensor)
        else:
            normalized = (tensor - min_val) / (range_val + self._eps)
            normalized = torch.clamp(normalized, 0, 1)
        normalized[~mask] = 255.
        return normalized

    def geometry(self, batch_data_samples, voxel_feats):
        geometric_loss = []
        slope = []
        unevenness = []
        height = []
        occupied_mask = []
        for i in range(len(batch_data_samples)):
            geom_occ_path = batch_data_samples[i].metainfo['geom_occ_path']
            _slope = self.normalize(np.load(geom_occ_path)['slope'])
            _unevenness = self.normalize(np.load(geom_occ_path)['unevenness'])
            _height = self.normalize(np.load(geom_occ_path)['height'])
            slope.append(_slope)
            unevenness.append(_unevenness)
            height.append(_height)
            occupied_mask.append((_slope != 255))
        slope = torch.stack(slope)
        unevenness = torch.stack(unevenness)
        height = torch.stack(height)
        occupied_mask = torch.stack(occupied_mask)
        _geometry = torch.stack([slope, unevenness, height],dim=-1).cuda()
        geometric_prediction = self.geometric_layer(voxel_feats[0].permute(0,2,3,4,1))
        if self.use_geometry == 'MSE':
            geometric_loss = self.geometric_loss(geometric_prediction[occupied_mask], _geometry[occupied_mask])
        elif self.use_geometry == 'CE':
            def make_LID_bins(n_bins, dmin, dmax):
                i = torch.arange(n_bins + 1, device=geometric_prediction[occupied_mask].device)
                return dmin + (dmax - dmin) * i * (i + 1) / (n_bins * (n_bins + 1))

            def depth_to_bin_index(depth, bin_edges):
                """
                depth: (N, 3)
                bin_edges: (n_bins + 1,)
                return: (N, 3) int64 bin indices
                """
                N, C = depth.shape
                flat = depth.view(-1)
                indices = torch.bucketize(flat, bin_edges, right=False) - 1
                return indices.view(N, C).clamp(min=0, max=len(bin_edges)-2)
            dmin, dmax = 1.0, self.n_bins
            bin_edges = make_LID_bins(self.n_bins, dmin, dmax)
            bin_gt = depth_to_bin_index(_geometry[occupied_mask], bin_edges)
            loss = 0
            for c in range(3):
                N,C = geometric_prediction[occupied_mask].shape
                logits = geometric_prediction[occupied_mask].view(N, 3, self.n_bins)[:, c, :]
                targets = bin_gt[:, c]
                loss += F.cross_entropy(logits, targets)
            geometric_loss = loss / 3
        return geometric_loss
    
    def safe_sigmoid(self, tensor):
        tensor = torch.clamp(tensor, -9.21, 9.21)
        return torch.sigmoid(tensor)
    def get_rotation_matrix(self, tensor):
        assert tensor.shape[-1] == 4

        tensor = F.normalize(tensor, dim=-1)
        mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
        mat1[..., 0, 0] = tensor[..., 0]
        mat1[..., 0, 1] = - tensor[..., 1]
        mat1[..., 0, 2] = - tensor[..., 2]
        mat1[..., 0, 3] = - tensor[..., 3]
        
        mat1[..., 1, 0] = tensor[..., 1]
        mat1[..., 1, 1] = tensor[..., 0]
        mat1[..., 1, 2] = - tensor[..., 3]
        mat1[..., 1, 3] = tensor[..., 2]

        mat1[..., 2, 0] = tensor[..., 2]
        mat1[..., 2, 1] = tensor[..., 3]
        mat1[..., 2, 2] = tensor[..., 0]
        mat1[..., 2, 3] = - tensor[..., 1]

        mat1[..., 3, 0] = tensor[..., 3]
        mat1[..., 3, 1] = - tensor[..., 2]
        mat1[..., 3, 2] = tensor[..., 1]
        mat1[..., 3, 3] = tensor[..., 0]

        mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
        mat2[..., 0, 0] = tensor[..., 0]
        mat2[..., 0, 1] = - tensor[..., 1]
        mat2[..., 0, 2] = - tensor[..., 2]
        mat2[..., 0, 3] = - tensor[..., 3]
        
        mat2[..., 1, 0] = tensor[..., 1]
        mat2[..., 1, 1] = tensor[..., 0]
        mat2[..., 1, 2] = tensor[..., 3]
        mat2[..., 1, 3] = - tensor[..., 2]

        mat2[..., 2, 0] = tensor[..., 2]
        mat2[..., 2, 1] = - tensor[..., 3]
        mat2[..., 2, 2] = tensor[..., 0]
        mat2[..., 2, 3] = tensor[..., 1]

        mat2[..., 3, 0] = tensor[..., 3]
        mat2[..., 3, 1] = tensor[..., 2]
        mat2[..., 3, 2] = - tensor[..., 1]
        mat2[..., 3, 3] = tensor[..., 0]

        mat2 = torch.conj(mat2).transpose(-1, -2)
        
        mat = torch.matmul(mat1, mat2)
        return mat[..., 1:, 1:]
    
    def k_means(self, label, mask, n_clusters=20, samples_per_cluster=100):
        total_samples = n_clusters * samples_per_cluster
        # 1. valid sample만 선택
        label = label[mask]  # (M, 3)
        label_np = label.detach().numpy()
        
        # 2. K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(label_np)
        cluster_labels = torch.tensor(kmeans.labels_, device=label.device)  # (M,)

        # 3. 클러스터별 인덱스 추출
        indices_per_cluster = [torch.where(cluster_labels == i)[0] for i in range(n_clusters)]

        # 4. 최소 개수 또는 사용자 지정 수 만큼 샘플링
        valid_clusters = [idx for idx in indices_per_cluster if len(idx) > 0]
        if samples_per_cluster is None:
            n_valid_clusters = len(valid_clusters)
            min_samples = min(len(idx) for idx in valid_clusters)
            total_possible = min_samples * n_valid_clusters
            if total_possible > total_samples:
                min_samples = total_samples // n_valid_clusters
        else:
            min_samples = samples_per_cluster

        sampled_cluster_indices = []
        for cluster_indices in valid_clusters:
            sample_count = min(len(cluster_indices), min_samples)
            perm = torch.randperm(len(cluster_indices))[:sample_count]
            sampled_cluster_indices.append(cluster_indices[perm])

        # 5. 전체 인덱스 변환 (mask 기반의 인덱싱이므로 global index로 복원 필요)
        valid_indices = mask.nonzero(as_tuple=True)[0]  # (M,)
        global_sampled_indices = torch.cat(sampled_cluster_indices)
        global_indices = valid_indices[global_sampled_indices]
        total_needed = total_samples
        if global_indices.shape[0] < total_needed:
            remaining = total_needed - global_indices.shape[0]
            all_valid_indices = mask.nonzero(as_tuple=True)[0]
            unused = torch.tensor(list(set(all_valid_indices.tolist()) - set(global_indices.tolist())), device=global_indices.device)
            if len(unused) >= remaining:
                extra = unused[torch.randperm(len(unused))[:remaining]]
            else:
                extra = unused  # 부족하면 남은 것만 사용
            global_indices = torch.cat([global_indices, extra])
        return global_indices.long()  # (K * min_samples,) tensor
    
    def gaussian_function(self, batch_inputs, batch_data_samples, voxel_feats, gaussian_logits, geometric_prediction):
        # gaussian_total_start = time.time()
        
        def gaussian_to_voxel_splatting_optimized(voxel_grid, gaussian_means, scales, rotation, opacities, gaussian_v):
            bs, g, _ = gaussian_means.shape
            S = torch.zeros(bs, g, 3, 3, dtype=gaussian_means.dtype, device=gaussian_means.device)
            S[..., 0, 0] = scales[..., 0]
            S[..., 1, 1] = scales[..., 1]
            S[..., 2, 2] = scales[..., 2]
            R = self.get_rotation_matrix(rotation) # b, g, 3, 3
            M = torch.matmul(S, R)
            Cov = torch.matmul(M.transpose(-1, -2), M)
            CovInv = Cov.float().cpu().inverse().cuda() # b, g, 3, 3
            semantics = self.aggregator(
                voxel_grid.clone().float(), 
                gaussian_means.float(), 
                opacities.float(),
                gaussian_v.float(),
                scales.float(),
                CovInv.float()) # 1, c, n
            return semantics

        def hybrid_voxel_splat_loss(voxel_features, voxel_grid, voxel_labels, geometric_label, gaussian_logits, batch_data_samples, geometric_logits, occ_trajectory):
            H, W, D = 256, 256, 32          # 현재 모델이 쓰는 최종 해상도

            # ── (2) voxel_grid 플랫 순서 그대로 인덱스 → (z,y,x) 정수 좌표 ──
            N = voxel_grid.size(0)          # (= voxel_features.size(0) = voxel_labels.size(0))
            idx = torch.arange(N, device=voxel_grid.device)
            z = idx // (W * D)
            y = (idx // D) % W
            x = idx % D
            voxel_coords = torch.stack([z, y, x], dim=1)   # (N,3)
            
            
            # Sampling
            label_mask = (voxel_labels.squeeze() == 1) | (voxel_labels.squeeze() == 2)
            coord_mask = (
                (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < H) &
                (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < W) &
                (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < D)
            )
            valid_mask = coord_mask & label_mask
            # random
            #valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            #sampled_idx = valid_indices[torch.randperm(len(valid_indices), device=valid_indices.device)[:10000]]
            # cluster
            sampled_idx = self.k_means(geometric_label, valid_mask.cpu())
            
            # Gaussian decoding 
            out = self.gaussian_decoder(voxel_features[sampled_idx])  # scale(3), rotation(4), opacity(1), logits(3), geometry(3)
            gs_scales = self.safe_sigmoid(out[:, :3]) 
            gs_scales = 0.01 + (1.6-0.01) * gs_scales # 가우시안의 크기를 조절 (8cm~32cm)
            rot = F.normalize(out[:, 3:7], 2, -1)
            opacities = self.safe_sigmoid(out[:,7])
            gaussian_v = gaussian_logits.view(-1,3)[sampled_idx]
            gaussian_v = gaussian_v.softmax(dim=-1)
            # Voxel splatting
            semantics = gaussian_to_voxel_splatting_optimized(
                voxel_grid=voxel_grid.unsqueeze(0),
                gaussian_means=voxel_grid[sampled_idx].unsqueeze(0),
                scales=gs_scales.unsqueeze(0),
                rotation=rot.unsqueeze(0),
                opacities=opacities.unsqueeze(0),
                gaussian_v=gaussian_v.unsqueeze(0),
            )
            no_render_mask = semantics[1] != 0
            pred = semantics[0]
            target = voxel_labels
            pred = torch.clamp(pred, 1e-6, 1. - 1e-6)
            loss = F.nll_loss(torch.log(pred)[no_render_mask], target[no_render_mask], ignore_index=255)
            range_mask = (target != 255) * no_render_mask
            lovasz_loss = lovasz_softmax(pred[range_mask], target[range_mask], ignore=0)
            return loss + 0.1 * lovasz_loss

        # Data preparation
        voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'], [1, 1, 1],
                                                  np.array([len(batch_data_samples), 256, 256, 32], dtype=np.int32))
        voxel_labels = voxels_lvl0.reshape(len(batch_data_samples), -1)

        B, C, X, Y, Z = voxel_feats[0].shape
        voxel_features = voxel_feats[0].permute(0, 2, 3, 4, 1).contiguous()
        voxel_features_flat = voxel_features.view(B, -1, C)
        
        pc_min = torch.tensor([-25.6, -12.8, -1.6])
        pc_max = torch.tensor([0.0, 12.8, 1.6])
        grid_size = (256, 256, 32)  # X, Y, Z
        voxe_size = torch.tensor(0.1)

        # 각 축에 대해 균일한 간격으로 좌표 생성
        grid_x = torch.linspace(pc_min[0] + voxe_size / 2, pc_max[0] - voxe_size / 2, grid_size[0])
        grid_y = torch.linspace(pc_min[1] + voxe_size / 2, pc_max[1] - voxe_size / 2, grid_size[1])
        grid_z = torch.linspace(pc_min[2] + voxe_size / 2, pc_max[2] - voxe_size / 2, grid_size[2])

        # 3D meshgrid 생성
        grid_xx, grid_yy, grid_zz = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

        # 좌표 스택
        voxel_grid = torch.stack([grid_xx, grid_yy, grid_zz], dim=-1).cuda()

        # (N, 3) 형태로 flatten
        voxel_grid_flat = voxel_grid.view(-1, 3)
        # Batch processing
        total_loss = 0
        for b in range(B):
            # Load geometry data
            geom_occ_path = batch_data_samples[b].metainfo['geom_occ_path']
            _slope = self.normalize(np.load(geom_occ_path)['slope'])
            _unevenness = self.normalize(np.load(geom_occ_path)['unevenness'])
            _height = self.normalize(np.load(geom_occ_path)['height'])
            geometric_labels = torch.stack([_slope, _height, _unevenness], dim=-1)

            features = voxel_features_flat[b]
            labels = voxel_labels[b].view(-1)
            geometric_label = geometric_labels.view(-1, 3)
            gaussian_logit = gaussian_logits[b]
            geometric_logit = False
            occ_trajectory = False
            loss = hybrid_voxel_splat_loss(features, voxel_grid_flat, labels, geometric_label, gaussian_logit, batch_data_samples, geometric_logit, occ_trajectory)
            total_loss += loss

        total_loss = total_loss / B
        # print(f"    [GAUSSIAN] TOTAL gaussian function: {(time.time() - gaussian_total_start)*1000:.2f}ms")
        return total_loss

    def _forward(self, batch_inputs, batch_data_samples): # img, img_metas, sparse_voxel, voxel_coords, batch_data_samples
        """Forward training function."""
        # forward_start = time.time()
        
        # 1. Image preprocessing
        imgs = batch_inputs['imgs']
        img_metas = []
        for data_sample in batch_data_samples:
            if not self.occ3d:
                img_meta = dict(lidar2img=data_sample.lidar2img,
                                img_shape=torch.Tensor([imgs.shape[-2],imgs.shape[-1]])
                                )
            else:
                img_meta = dict(lidar2img=data_sample.ego2img,
                                img_shape=torch.Tensor([imgs.shape[-2],imgs.shape[-1]])
                                )
            img_metas.append(img_meta)
        
        # 2. Feature extraction
        img_feats = self.extract_feat(imgs)
        
        # 3. LiDAR/Radar processing
        if self.use_lidar:
            lidar_xyz_feat = self.svfe_lidar(batch_inputs['lidar_voxel_feats'], batch_inputs['lidar_voxel_coords']) # [B, C, X, Y, Z]
        if self.use_radar:
            radar_xyz_feat = self.svfe_radar(batch_inputs['radar_voxel_feats'], batch_inputs['radar_voxel_coords'])
        
        # 4. View transformation
        if (not self.use_lidar) and (not self.use_radar):
            xyz_volumes = self.view_transformer(img_feats, img_metas) # [B, C, X, Y, Z]
        elif self.use_lidar and (not self.use_radar):
            xyz_volumes = self.view_transformer.forward_two(img_feats, img_metas, lidar_xyz_feat) #  lidar_xyz_feat
        elif (not self.use_lidar) and self.use_radar:
            xyz_volumes = self.view_transformer.forward_two(img_feats, img_metas, radar_xyz_feat)
        elif self.use_lidar and self.use_radar:
            xyz_volumes = self.view_transformer.forward_three(img_feats, img_metas, lidar_xyz_feat, radar_xyz_feat)
        # 6. Geometry loss (if enabled)
        if self.use_geometry and type(self.use_geometry) != bool and xyz_volumes[0].requires_grad:
            geometric_loss = self.geometry(batch_data_samples, xyz_volumes)
            # 7. Occupancy head
            occ_result = self.occ_head(xyz_volumes)
            # print(f"[DEBUG] TOTAL _forward time: {(time.time() - forward_start)*1000:.2f}ms\n")
            return occ_result, geometric_loss
        # 7. Occupancy head (without geometry)
        occ_result = self.occ_head(xyz_volumes)
        # 5. Gaussian loss computation
        if self.use_gaussian and xyz_volumes[0].requires_grad:
            geometric_prediction=False
            gaussian_loss = self.gaussian_function(batch_inputs, batch_data_samples, xyz_volumes, occ_result[0], geometric_prediction)
        else:
            gaussian_loss = False
        # print(f"[DEBUG] TOTAL _forward time: {(time.time() - forward_start)*1000:.2f}ms\n")
        return occ_result, False, gaussian_loss
             
    def uncertainty(self, probs, num_c):
        # Use cached log computation
        entropy_map = -torch.sum(probs * torch.log(probs + self._eps), dim=-1)
        entropy_map = 2 - entropy_map / self._log_tensor_cache.get('log_3', 
                                      self._log_tensor_cache.setdefault('log_3', torch.log(torch.tensor(3.0, device=probs.device))))
        return entropy_map.view(-1).detach()
        
    def loss(self, batch_inputs, batch_data_samples):
        loss_total_start = time.time()
        
        # Forward pass
        vox_logits, geometric_loss, gaussian_loss = self._forward(batch_inputs,batch_data_samples)
        
        # Process outputs
        vox_logits_lvl0, vox_logits_lvl1, vox_logits_lvl2, vox_logits_lvl3 = vox_logits
        B,X,Y,Z,Cls = vox_logits_lvl0.shape
        
        # Multiscale supervision
        if X==256:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[2,2,2],np.array([len(batch_data_samples),128,128,16],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[4,4,4],np.array([len(batch_data_samples),64,64,8],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[8,8,8],np.array([len(batch_data_samples),32,32,4],dtype=np.int32)) 
        elif self.occ3d:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_3d_masked'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['occ_3d'],[2,2,2],np.array([len(batch_data_samples),100,100,8],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['occ_3d'],[4,4,4],np.array([len(batch_data_samples),50,50,4],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['occ_3d'],[8,8,8],np.array([len(batch_data_samples),25,25,2],dtype=np.int32)) 
        else:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[2,2,2],np.array([len(batch_data_samples),100,100,8],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[4,4,4],np.array([len(batch_data_samples),50,50,4],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[8,8,8],np.array([len(batch_data_samples),25,25,2],dtype=np.int32))
        
        # Data preparation for loss computation (optimized with batch operations)
        # Level 0
        vox_fl_predict_lvl0 = F.softmax(vox_logits_lvl0.view(B, -1, Cls), dim=-1)
        vox_fl_label_lvl0 = voxels_lvl0.view(B, -1)
        vox_sem_predict_lvl0 = vox_logits_lvl0.permute(0, 4, 1, 2, 3)
        vox_lovasz_lvl0 = vox_logits_lvl0.view(-1, Cls)
        vox_lovasz_label_lvl0 = voxels_lvl0.view(-1)

        # Level 1
        vox_fl_predict_lvl1 = F.softmax(vox_logits_lvl1.view(B, -1, Cls), dim=-1)
        vox_fl_label_lvl1 = voxels_lvl1.view(B, -1)
        vox_sem_predict_lvl1 = vox_logits_lvl1.permute(0, 4, 1, 2, 3)
        vox_lovasz_lvl1 = vox_logits_lvl1.view(-1, Cls)
        vox_lovasz_label_lvl1 = voxels_lvl1.view(-1)
        
        # Level 2
        vox_fl_predict_lvl2 = F.softmax(vox_logits_lvl2.view(B, -1, Cls), dim=-1)
        vox_fl_label_lvl2 = voxels_lvl2.view(B, -1)
        vox_sem_predict_lvl2 = vox_logits_lvl2.permute(0, 4, 1, 2, 3)
        vox_lovasz_lvl2 = vox_logits_lvl2.view(-1, Cls)
        vox_lovasz_label_lvl2 = voxels_lvl2.view(-1)
        
        # Level 3
        vox_fl_predict_lvl3 = F.softmax(vox_logits_lvl3.view(B, -1, Cls), dim=-1)
        vox_fl_label_lvl3 = voxels_lvl3.view(B, -1)
        vox_sem_predict_lvl3 = vox_logits_lvl3.permute(0, 4, 1, 2, 3)
        vox_lovasz_lvl3 = vox_logits_lvl3.view(-1, Cls)
        vox_lovasz_label_lvl3 = voxels_lvl3.view(-1)

        # Uncertainty computation
        if self.use_uncertainty:
            weight_0 = self.uncertainty(vox_fl_predict_lvl0, Cls)
            weight_1 = self.uncertainty(vox_fl_predict_lvl1, Cls)
            weight_2 = self.uncertainty(vox_fl_predict_lvl2, Cls)
            weight_3 = self.uncertainty(vox_fl_predict_lvl3, Cls)
        else:
            weight_0 = weight_1 = weight_2 = weight_3 = 1
        
        # Optimized loss computation with reduced function calls
        def compute_level_loss(fl_pred, fl_label, sem_pred, voxels, lovasz_pred, lovasz_label, weight, scale=1.0):
            fl_loss = self.loss_fl(fl_pred, fl_label, weight)
            geo_loss = geo_scal_loss(sem_pred, voxels)
            sem_loss = sem_scal_loss(sem_pred, voxels)
            lovasz_loss = self.loss_lovasz(lovasz_pred, lovasz_label)
            
            # Use torch.stack for efficient nan handling
            losses = torch.stack([fl_loss, geo_loss, sem_loss, lovasz_loss])
            losses = torch.nan_to_num(losses)
            return scale * losses.sum()
        
        loss = dict(
            level0_loss=compute_level_loss(vox_fl_predict_lvl0, vox_fl_label_lvl0, vox_sem_predict_lvl0, 
                                         voxels_lvl0, vox_lovasz_lvl0, vox_lovasz_label_lvl0, weight_0, 1.0),
            level1_loss=compute_level_loss(vox_fl_predict_lvl1, vox_fl_label_lvl1, vox_sem_predict_lvl1, 
                                         voxels_lvl1, vox_lovasz_lvl1, vox_lovasz_label_lvl1, weight_1, 0.5),
            level2_loss=compute_level_loss(vox_fl_predict_lvl2, vox_fl_label_lvl2, vox_sem_predict_lvl2, 
                                         voxels_lvl2, vox_lovasz_lvl2, vox_lovasz_label_lvl2, weight_2, 0.25),
            level3_loss=compute_level_loss(vox_fl_predict_lvl3, vox_fl_label_lvl3, vox_sem_predict_lvl3, 
                                         voxels_lvl3, vox_lovasz_lvl3, vox_lovasz_label_lvl3, weight_3, 0.125)
        )
        
        # Add additional losses
        if type(geometric_loss) != bool:
            loss['geometric_loss'] = geometric_loss
        if type(gaussian_loss) != bool:
            loss['gaussian_loss'] = gaussian_loss
        return loss
    
    # def predict(self, batch_inputs,batch_data_samples):
    #     """Forward predict function."""
    #     with torch.no_grad():
    #         occ_ori_logits , _, _= self._forward(batch_inputs,batch_data_samples)
    #     B,X,Y,Z,Cls = occ_ori_logits.shape
    #     if X==256:
    #         voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
    #         voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
    #         #import pdb; pdb.set_trace()
    #         #voxels_lvl0_np = voxels_lvl0.cpu().numpy()
    #         #np.save('voxels_gt.npy', voxels_lvl0_np)
    #     elif self.occ3d:
    #         voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_3d_masked'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
    #         voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
    #     else:
    #         voxels_lvl0 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
    #         voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
            
    #     for i, data_sample in enumerate(batch_data_samples):
    #         data_sample.eval_ann_info['pts_semantic_mask'] = voxels_lvl0[i].cpu().numpy().astype(np.uint8)

        
    #     final_vox_logits = [occ_ori_logit.reshape(-1,Cls) for occ_ori_logit in occ_ori_logits]
    #     return self.postprocess_result(final_vox_logits, batch_data_samples)
    
    def predict(self, batch_inputs, batch_data_samples):
        """Forward predict function with minimal RAM/GPU usage."""
        # ① Disable autograd
        with torch.no_grad():
            occ_logits, _, _ = self._forward(batch_inputs, batch_data_samples)  # (B,X,Y,Z,C)

        # ② Keep only argmax mask on CPU
        occ_preds = occ_logits.argmax(-1).to(torch.uint8).cpu()  # (B,X,Y,Z)
        # free GPU & CPU refs
        #import pdb;pdb.set_trace()

        # ③ Prepare GT for metric (same logic as original)
        B, X, Y, Z = occ_preds.shape
        if X == 256:
            gt_lvl0 = self.multiscale_supervision(
                batch_inputs['occ_semantickitti_masked'], [1, 1, 1],
                np.array([len(batch_data_samples), 256, 256, 32], np.int32))
        elif self.occ3d:
            gt_lvl0 = self.multiscale_supervision(
                batch_inputs['occ_3d_masked'], [1, 1, 1],
                np.array([len(batch_data_samples), 200, 200, 16], np.int32))
        else:
            gt_lvl0 = self.multiscale_supervision(
                batch_inputs['dense_occ_200'], [1, 1, 1],
                np.array([len(batch_data_samples), 200, 200, 16], np.int32))
        gt_lvl0 = gt_lvl0.reshape(len(batch_data_samples), -1)

        # ④ Populate DataSample with CPU mask only
        for i, ds in enumerate(batch_data_samples):
            ds.eval_ann_info['pts_semantic_mask'] = gt_lvl0[i].cpu().numpy().astype(np.uint8)
            ds.set_data({
            'pred_pts_seg': PointData(pts_semantic_mask=occ_preds[i].reshape(-1).numpy())
        })

            # Optional: save prediction mask to disk
            if getattr(self, 'npy_save_folder', None):
                meta = ds.metainfo
                sample_id = meta.get('sample_id', f"{i:05d}")
                frame_id  = meta.get('frame_id', f"{i:06d}")
                out_path = os.path.join(self.npy_save_folder,
                        f"{frame_id}.label")
                (occ_preds[i]              # (X,Y,Z)
                .reshape(-1)            # -> (N,)
                .numpy()
                .astype(np.uint8)
                .tofile(out_path))  
                print(f"[i] Saved logits to {out_path}")
                del occ_logits 
        return batch_data_samples
    
    def postprocess_result(self, voxel_logits, batch_data_samples):
        #import pdb; pdb.set_trace()
        # for i in range(len(voxel_logits)):
        #     voxel_logit = voxel_logits[i]
        #     voxel_pred = voxel_logit.argmax(dim=1)

        #     if self.npy_save_folder:
        #         # batch_data_samples의 metainfo에서 sample_id와 frame_id 추출
        #         metainfo = batch_data_samples[i].metainfo
        #         sample_id = metainfo.get('sample_id', f"{i:05d}")
        #         frame_id = metainfo.get('frame_id', f"{i:06d}")
        #         npy_filename = f"{sample_id}_{frame_id}.npy"
        #         npy_filepath = os.path.join(self.npy_save_folder, npy_filename)
        #         np.save(npy_filepath, voxel_logit.cpu().numpy())
        #         print(f"Saved voxel logit to {npy_filepath}")

        #     batch_data_samples[i].set_data({
        #         'pts_seg_logits':
        #         PointData(**{'pts_seg_logits': voxel_logit}),
        #         'pred_pts_seg':
        #         PointData(**{'pts_semantic_mask': voxel_pred})
        #     })
        return batch_data_samples
    
    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs, batch_data_samples):
        pass
    


@MODELS.register_module()
class TraversFusion(Base3DSegmentor):
    def __init__(self,
                 use_occ3d,
                 use_lidar,
                 use_radar,
                 data_preprocessor,
                 backbone,
                 neck,
                 view_transformer,
                 svfe_lidar,
                 svfe_radar,
                 occ_head):
        super().__init__(data_preprocessor=data_preprocessor)
        self.occ3d = use_occ3d
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.img_backbone = MODELS.build(backbone)
        self.img_neck = MODELS.build(neck)
        if self.use_lidar:
            self.svfe_lidar = MODELS.build(svfe_lidar)
        self.view_transformer = MODELS.build(view_transformer)
        self.occ_head = MODELS.build(occ_head)
        self.loss_fl = FocalLoss(gamma=2,ignore_index=255) # 0: noise label weights=
        self.loss_lovasz = LovaszLoss(loss_type='multi_class',
                                      per_sample=False,
                                      reduction='none')
        self.temporal_list = []
        self.fusion_conv = nn.ModuleList()
        for i in range(len(self.occ_head.channels)):
            mlp_occ = nn.Sequential(
                nn.Linear(self.occ_head.channels[i]*2, self.occ_head.channels[i]),
                nn.ReLU(),
                nn.Linear(self.occ_head.channels[i], self.occ_head.channels[i]),
                nn.ReLU(),
                nn.Linear(self.occ_head.channels[i], self.occ_head.channels[i])
            )
            self.fusion_conv.append(mlp_occ)

    def multiscale_supervision(self, gt_occ, ratio, gt_shape):
        gt = torch.zeros([gt_shape[0], gt_shape[1], gt_shape[2], gt_shape[3]]).to(gt_occ[0].device).type(torch.long) 
        for i in range(gt.shape[0]):
            coords_x = gt_occ[i][:, 0].to(torch.float) // ratio[0]
            coords_y = gt_occ[i][:, 1].to(torch.float) // ratio[1]
            coords_z = gt_occ[i][:, 2].to(torch.float) // ratio[2]
            coords_x = coords_x.to(torch.long)
            coords_y = coords_y.to(torch.long)
            coords_z = coords_z.to(torch.long)
            coords = torch.stack([coords_x,coords_y,coords_z],dim=1)
            gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[i][:, 3]

        return gt
    
    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped    

    
    def _forward(self, batch_inputs, batch_data_samples): # img, img_metas, sparse_voxel, voxel_coords, batch_data_samples
        """Forward training function."""
        imgs = batch_inputs['imgs']
        img_metas = []
        for data_sample in batch_data_samples:
            if not self.occ3d:
                img_meta = dict(lidar2img=data_sample.lidar2img,
                                img_shape=torch.Tensor([imgs.shape[-2],imgs.shape[-1]])
                                )
            else:
                img_meta = dict(lidar2img=data_sample.ego2img,
                                img_shape=torch.Tensor([imgs.shape[-2],imgs.shape[-1]])
                                )
            img_metas.append(img_meta)
        img_feats = self.extract_feat(imgs)
        if self.use_lidar:
            lidar_xyz_feat = self.svfe_lidar(batch_inputs['lidar_voxel_feats'], batch_inputs['lidar_voxel_coords']) # [B, C, X, Y, Z]
        if self.use_lidar and (not self.use_radar):
            xyz_volumes = self.view_transformer.forward_two(img_feats, img_metas, lidar_xyz_feat) #  lidar_xyz_feat
        return self.occ_head(xyz_volumes) # fused_xyz_feat
             
    def loss(self, batch_inputs, batch_data_samples):
        logits, uncertainty = self._forward(batch_inputs,batch_data_samples)
        vox_logits_lvl0, vox_logits_lvl1, vox_logits_lvl2, vox_logits_lvl3 = logits
        vox_uncertainty_lvl0, vox_uncertainty_lvl1, vox_uncertainty_lvl2, vox_uncertainty_lvl3 = uncertainty
        B,X,Y,Z,Cls = vox_logits_lvl0.shape
        if X==256:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[2,2,2],np.array([len(batch_data_samples),128,128,16],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[4,4,4],np.array([len(batch_data_samples),64,64,8],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[8,8,8],np.array([len(batch_data_samples),32,32,4],dtype=np.int32)) 
        elif self.occ3d:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_3d_masked'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['occ_3d'],[2,2,2],np.array([len(batch_data_samples),100,100,8],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['occ_3d'],[4,4,4],np.array([len(batch_data_samples),50,50,4],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['occ_3d'],[8,8,8],np.array([len(batch_data_samples),25,25,2],dtype=np.int32))
        else:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl1 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[2,2,2],np.array([len(batch_data_samples),100,100,8],dtype=np.int32))
            voxels_lvl2 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[4,4,4],np.array([len(batch_data_samples),50,50,4],dtype=np.int32))
            voxels_lvl3 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[8,8,8],np.array([len(batch_data_samples),25,25,2],dtype=np.int32))
            
        vox_fl_predict_lvl0 = vox_logits_lvl0.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl0 = vox_logits_lvl0.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl0 = vox_logits_lvl0.reshape(-1,Cls)
        vox_lovasz_label_lvl0 = voxels_lvl0.reshape(-1)
        vox_ce_uncertainty_lvl0 = vox_uncertainty_lvl0.reshape(B,-1,1) # [Bs,Num,Cls]

        vox_fl_predict_lvl1 = vox_logits_lvl1.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl1 = voxels_lvl1.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl1 = vox_logits_lvl1.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl1 = vox_logits_lvl1.reshape(-1,Cls)
        vox_lovasz_label_lvl1 = voxels_lvl1.reshape(-1)
        vox_ce_uncertainty_lvl1 = vox_uncertainty_lvl1.reshape(B,-1,1) # [Bs,Num,Cls]
        
        vox_fl_predict_lvl2 = vox_logits_lvl2.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl2 = voxels_lvl2.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl2 = vox_logits_lvl2.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl2 = vox_logits_lvl2.reshape(-1,Cls)
        vox_lovasz_label_lvl2 = voxels_lvl2.reshape(-1)
        vox_ce_uncertainty_lvl2 = vox_uncertainty_lvl2.reshape(B,-1,1) # [Bs,Num,Cls]
        
        vox_fl_predict_lvl3 = vox_logits_lvl3.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl3 = voxels_lvl3.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl3 = vox_logits_lvl3.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl3 = vox_logits_lvl3.reshape(-1,Cls)
        vox_lovasz_label_lvl3 = voxels_lvl3.reshape(-1)
        vox_ce_uncertainty_lvl3 = vox_uncertainty_lvl3.reshape(B,-1,1) # [Bs,Num,Cls]
        loss = dict(level0_loss = torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl0,vox_fl_label_lvl0)) + \
                                  torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl0,voxels_lvl0)) + \
                                  torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl0,voxels_lvl0)) + \
                                  torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl0,vox_lovasz_label_lvl0)) + \
                                  torch.nan_to_num(elbo_loss(vox_logits_lvl0.view(-1,4).softmax(-1),voxels_lvl0.view(-1), vox_uncertainty_lvl0.view(-1,1))),
                    level1_loss = 0.5 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl1,vox_fl_label_lvl1)) + \
                                        torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl1,voxels_lvl1)) + \
                                        torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl1,voxels_lvl1)) + \
                                        torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl1,vox_lovasz_label_lvl1))) + \
                                        torch.nan_to_num(elbo_loss(vox_logits_lvl1.view(-1,4).softmax(-1),voxels_lvl1.view(-1), vox_uncertainty_lvl1.view(-1,1))),
                    level2_loss = 0.25 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl2,vox_fl_label_lvl2)) + \
                                          torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl2,voxels_lvl2)) + \
                                          torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl2,voxels_lvl2)) + \
                                          torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl2,vox_lovasz_label_lvl2))) + \
                                          torch.nan_to_num(elbo_loss(vox_logits_lvl2.view(-1,4).softmax(-1),voxels_lvl2.view(-1), vox_uncertainty_lvl2.view(-1,1))),
                    level3_loss = 0.125 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl3,vox_fl_label_lvl3)) + \
                                           torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl3,voxels_lvl3)) + \
                                           torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl3,voxels_lvl3)) + \
                                           torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl3,vox_lovasz_label_lvl3))) + \
                                           torch.nan_to_num(elbo_loss(vox_logits_lvl3.view(-1,4).softmax(-1),voxels_lvl3.view(-1), vox_uncertainty_lvl3.view(-1,1)))
                    )
        return loss
    
    def predict(self, batch_inputs,batch_data_samples):
        """Forward predict function."""
        occ_ori_logits = self._forward(batch_inputs,batch_data_samples)
        B,X,Y,Z,Cls = occ_ori_logits.shape
        if X==256:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
        elif self.occ3d:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_3d_masked'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
        else:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
            
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.eval_ann_info['pts_semantic_mask'] = voxels_lvl0[i].cpu().numpy().astype(np.uint8)

        final_vox_logits = [occ_ori_logit.reshape(-1,Cls) for occ_ori_logit in occ_ori_logits]
        return self.postprocess_result(final_vox_logits, batch_data_samples)
    
    def postprocess_result(self, voxel_logits, batch_data_samples):
        
        for i in range(len(voxel_logits)):
            
            voxel_logit = voxel_logits[i]
            voxel_pred = voxel_logit.argmax(dim=1)
            batch_data_samples[i].set_data({
                'pts_seg_logits':
                PointData(**{'pts_seg_logits': voxel_logit}),
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': voxel_pred})
            })
        return batch_data_samples
    
    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs, batch_data_samples):
        pass
    