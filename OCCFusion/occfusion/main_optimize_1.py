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
from .loss import geo_scal_loss, sem_scal_loss, elbo_loss
import cv2
from torch.nn import MSELoss
import random
import faiss
import torch_scatter
import time
from tqdm import tqdm
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
        self.gaussian_decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 10 + 3 - 3 + 3)  # mean(3), scale(3), rotation(4), logits(C) - mean(3) + geometry(3)
        )
        
        # Pre-initialize FAISS resources to avoid repeated setup overhead
        self.faiss_resources = None
        self._init_faiss_resources()

    def _init_faiss_resources(self):
        """Initialize FAISS resources once to avoid repeated overhead"""
        try:
            self.faiss_resources = faiss.StandardGpuResources()
        except Exception as e:
            print(f"Warning: Could not initialize FAISS GPU resources: {e}")
            self.faiss_resources = None

    def multiscale_supervision(self, gt_occ, ratio, gt_shape):
        #gt = torch.zeros([gt_shape[0], gt_shape[1], gt_shape[2], gt_shape[3]]).to(gt_occ[0].device).type(torch.long) 
        gt = torch.full(
        [gt_shape[0], gt_shape[1], gt_shape[2], gt_shape[3]],
        255,
        dtype=torch.long,
        device=gt_occ[0].device
        )
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
        tensor = torch.tensor(array).to(torch.float32)
        mask = tensor != 255
        valid = tensor[mask]
        min_val = valid.min()
        max_val = valid.max()
        normalized = (tensor - min_val) / (max_val - min_val + 1e-8)
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
    
    
    def gaussian(self, batch_inputs, batch_data_samples, voxel_feats):
        # gaussian_total_start = time.time()
        
        def gaussian_to_voxel_splatting_optimized(voxel_grid, gaussian_means, gaussian_covs, gaussian_logits, gaussian_geometry, voxel_size):
            V, G = voxel_grid.shape[0], gaussian_means.shape[0]
            
            voxel_logits = torch.zeros((V, gaussian_logits.shape[1] + gaussian_geometry.shape[1]), device=gaussian_logits.device)
            topk = 5

            # Memory-efficient chunked computation for large datasets
            chunk_size = 8192  # Adjust based on available memory
            if V * G > chunk_size * chunk_size:
                indices_list = []
                for i in range(0, V, chunk_size):
                    end_i = min(i + chunk_size, V)
                    voxel_chunk = voxel_grid[i:end_i]
                    
                    # Compute distances for this chunk
                    diff_chunk = voxel_chunk.unsqueeze(1) - gaussian_means.unsqueeze(0)  # [chunk, G, 3]
                    distances_chunk = torch.sum(diff_chunk ** 2, dim=-1)  # [chunk, G]
                    
                    # Get top-k for this chunk
                    _, indices_chunk = torch.topk(distances_chunk, k=min(topk, G), dim=1, largest=False)
                    indices_list.append(indices_chunk)
                
                indices = torch.cat(indices_list, dim=0)
            else:
                # Standard computation for smaller datasets
                diff = voxel_grid.unsqueeze(1) - gaussian_means.unsqueeze(0)  # [V, G, 3]
                distances = torch.sum(diff ** 2, dim=-1)  # [V, G]
                _, indices = torch.topk(distances, k=min(topk, G), dim=1, largest=False)  # [V, k]
            
            # Flatten indices for efficient indexing
            voxel_idx = torch.arange(V, device=voxel_grid.device).unsqueeze(1).repeat(1, min(topk, G)).view(-1)
            gaussian_idx = indices.view(-1)

            # Compute weights using gathered differences
            diff_selected = voxel_grid[voxel_idx] - gaussian_means[gaussian_idx]
            cov_diag = gaussian_covs[gaussian_idx].squeeze()
            cov_inv_diag = 1.0 / (cov_diag.diagonal(dim1=-2, dim2=-1) + 1e-6)

            mahalanobis = torch.sum(diff_selected ** 2 * cov_inv_diag, dim=-1)
            weight = torch.exp(-0.5 * mahalanobis)

            class_logits = torch.cat([gaussian_logits, gaussian_geometry], dim=-1)
            weighted_logits = weight.unsqueeze(-1) * class_logits[gaussian_idx]

            voxel_logits = torch_scatter.scatter_add(weighted_logits, voxel_idx, dim=0, out=voxel_logits)
            
            return voxel_logits

        def hybrid_voxel_splat_loss(voxel_features, voxel_grid, voxel_labels, geometric_label):
            # Sampling
            valid_mask = (voxel_labels.squeeze() != 255)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            sampled_idx = valid_indices[torch.randperm(len(valid_indices))[:16000]]

            # Gaussian decoding
            out = self.gaussian_decoder(voxel_features[sampled_idx])
            gaussian_scales = F.softplus(out[:, :3]) + 1e-2
            gaussian_logits = out[:, 7:10]
            gaussian_geometry = out[:, 10:]
            gaussian_covs = torch.diag_embed(gaussian_scales ** 2)

            # Voxel splatting
            voxel_logits = gaussian_to_voxel_splatting_optimized(
                voxel_grid=voxel_grid,
                gaussian_means=voxel_grid[sampled_idx],
                gaussian_covs=gaussian_covs,
                gaussian_logits=gaussian_logits,
                gaussian_geometry=gaussian_geometry,
                voxel_size=1.0
            )

            # Loss computation
            semantic_loss = F.cross_entropy(voxel_logits[:, :3], voxel_labels, ignore_index=255)
            geometry_loss = F.smooth_l1_loss(voxel_logits[:, 3:][valid_mask], geometric_label.cuda()[valid_mask.cpu()])
            
            return semantic_loss + geometry_loss

        # Data preparation
        voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'], [1, 1, 1],
                                                  np.array([len(batch_data_samples), 256, 256, 32], dtype=np.int32))
        voxel_labels = voxels_lvl0.reshape(len(batch_data_samples), -1)

        B, C, X, Y, Z = voxel_feats[0].shape
        voxel_features = voxel_feats[0].permute(0, 2, 3, 4, 1).contiguous()
        voxel_features_flat = voxel_features.view(B, -1, C)

        grid_x = torch.arange(X).view(1, X, 1, 1, 1).expand(B, X, Y, Z, 1)
        grid_y = torch.arange(Y).view(1, 1, Y, 1, 1).expand(B, X, Y, Z, 1)
        grid_z = torch.arange(Z).view(1, 1, 1, Z, 1).expand(B, X, Y, Z, 1)
        voxel_centers = torch.cat([grid_x, grid_y, grid_z], dim=-1).float().to(voxel_features.device)
        voxel_centers_flat = voxel_centers.view(B, -1, 3)

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
            voxel_grid = voxel_centers_flat[b]
            labels = voxel_labels[b].view(-1)
            geometric_label = geometric_labels.view(-1, 3)

            loss = hybrid_voxel_splat_loss(features, voxel_grid, labels, geometric_label)
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
        
        # 5. Gaussian loss computation
        # gaussian_loss = self.gaussian(batch_inputs, batch_data_samples, xyz_volumes)
        gaussian_loss = False
        # 6. Geometry loss (if enabled)
        if self.use_geometry and type(self.use_geometry) != bool and xyz_volumes[0].requires_grad:
            geometric_loss = self.geometry(batch_data_samples, xyz_volumes)
            
            # 7. Occupancy head
            occ_result = self.occ_head(xyz_volumes)
            # print(f"[DEBUG] TOTAL _forward time: {(time.time() - forward_start)*1000:.2f}ms\n")
            return occ_result, geometric_loss
        
        # 7. Occupancy head (without geometry)
        occ_result = self.occ_head(xyz_volumes)
        # print(f"[DEBUG] TOTAL _forward time: {(time.time() - forward_start)*1000:.2f}ms\n")
        return occ_result, False, gaussian_loss
             
    def uncertainty(self, probs, num_c):
        entropy_map = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B, H, W)
        entropy_map = 2-entropy_map/torch.log(torch.tensor(3))
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
        
        # Data preparation for loss computation
        vox_fl_predict_lvl0 = vox_logits_lvl0.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl0 = vox_logits_lvl0.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl0 = vox_logits_lvl0.reshape(-1,Cls)
        vox_lovasz_label_lvl0 = voxels_lvl0.reshape(-1)

        vox_fl_predict_lvl1 = vox_logits_lvl1.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl1 = voxels_lvl1.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl1 = vox_logits_lvl1.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl1 = vox_logits_lvl1.reshape(-1,Cls)
        vox_lovasz_label_lvl1 = voxels_lvl1.reshape(-1)
        
        vox_fl_predict_lvl2 = vox_logits_lvl2.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl2 = voxels_lvl2.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl2 = vox_logits_lvl2.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl2 = vox_logits_lvl2.reshape(-1,Cls)
        vox_lovasz_label_lvl2 = voxels_lvl2.reshape(-1)
        
        vox_fl_predict_lvl3 = vox_logits_lvl3.reshape(B,-1,Cls).softmax(dim=-1) # [Bs,Num,Cls]
        vox_fl_label_lvl3 = voxels_lvl3.reshape(len(batch_data_samples),-1) # [Bs,Num] 
        vox_sem_predict_lvl3 = vox_logits_lvl3.permute(0,4,1,2,3) # [Bs,Cls,X,Y,Z]
        vox_lovasz_lvl3 = vox_logits_lvl3.reshape(-1,Cls)
        vox_lovasz_label_lvl3 = voxels_lvl3.reshape(-1)

        # Uncertainty computation
        if self.use_uncertainty:
            weight_0 = self.uncertainty(vox_fl_predict_lvl0, Cls)
            weight_1 = self.uncertainty(vox_fl_predict_lvl1, Cls)
            weight_2 = self.uncertainty(vox_fl_predict_lvl2, Cls)
            weight_3 = self.uncertainty(vox_fl_predict_lvl3, Cls)
        else:
            weight_0 = weight_1 = weight_2 = weight_3 = 1
        
        # Loss computation
        loss = dict(level0_loss = torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl0,vox_fl_label_lvl0, weight_0)) + \
                                torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl0,voxels_lvl0)) + \
                                torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl0,voxels_lvl0)) + \
                                torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl0,vox_lovasz_label_lvl0)),
                    level1_loss = 0.5 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl1,vox_fl_label_lvl1, weight_1)) + \
                                        torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl1,voxels_lvl1)) + \
                                        torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl1,voxels_lvl1)) + \
                                        torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl1,vox_lovasz_label_lvl1))),
                    level2_loss = 0.25 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl2,vox_fl_label_lvl2, weight_2)) + \
                                        torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl2,voxels_lvl2)) + \
                                        torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl2,voxels_lvl2)) + \
                                        torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl2,vox_lovasz_label_lvl2))),
                    level3_loss = 0.125 * (torch.nan_to_num(self.loss_fl(vox_fl_predict_lvl3,vox_fl_label_lvl3, weight_3)) + \
                                        torch.nan_to_num(geo_scal_loss(vox_sem_predict_lvl3,voxels_lvl3)) + \
                                        torch.nan_to_num(sem_scal_loss(vox_sem_predict_lvl3,voxels_lvl3)) + \
                                        torch.nan_to_num(self.loss_lovasz(vox_lovasz_lvl3,vox_lovasz_label_lvl3)))
                    )
        
        # Add additional losses
        if type(geometric_loss) != bool:
            loss['geometric_loss'] = geometric_loss
        if type(gaussian_loss) != bool:
            loss['gaussian_loss'] = gaussian_loss
        # print(f"[LOSS] TOTAL loss function: {(time.time() - loss_total_start)*1000:.2f}ms")
        # print("="*80)
        return loss
    
    def predict(self, batch_inputs,batch_data_samples):
        """Forward predict function."""
        occ_ori_logits , _= self._forward(batch_inputs,batch_data_samples)
        B,X,Y,Z,Cls = occ_ori_logits.shape
        if X==256:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
            #import pdb; pdb.set_trace()
            #voxels_lvl0_np = voxels_lvl0.cpu().numpy()
            #np.save('voxels_gt.npy', voxels_lvl0_np)
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
        #import pdb; pdb.set_trace()
        for i in range(len(voxel_logits)):
            voxel_logit = voxel_logits[i]
            voxel_pred = voxel_logit.argmax(dim=1)

            if self.npy_save_folder:
                # batch_data_samples의 metainfo에서 sample_id와 frame_id 추출
                metainfo = batch_data_samples[i].metainfo
                sample_id = metainfo.get('sample_id', f"{i:05d}")
                frame_id = metainfo.get('frame_id', f"{i:06d}")
                npy_filename = f"{sample_id}_{frame_id}.npy"
                npy_filepath = os.path.join(self.npy_save_folder, npy_filename)
                np.save(npy_filepath, voxel_logit.cpu().numpy())
                print(f"Saved voxel logit to {npy_filepath}")

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

    def temporal_fusion(self, xyz_volumes, batch_data_samples):
        pose_file = os.path.join("/".join(batch_data_samples[0].img_path.split("/")[:-2]),"poses.txt")
        pose_file = open(pose_file, "r").read().split("\n")

        index = int(batch_data_samples[0].lidar_path.split("/")[-1][:-4])
        pose_4x4 = np.concatenate((np.array(pose_file[index].split(" ")).astype("float").reshape(3,4),np.array([[0,0,0,1]])))

        """
        pose_4x4 = sample["pose"] -> pose_file = os.path.join(args.data_root, "Rellis-3D", seq_str, "poses.txt") 로부터 얻은 각 시점별 에고 pose 정보
        homo_local = np.hstack((pc_local, np.ones((pc_local.shape[0], 1))))
        pc_world = (pose_4x4 @ homo_local.T).T[:, :3] -> 이 코드를 통해 라이다 좌표계의 pc를 월드 좌표계로 변환

        window_points_world = np.vstack(points_world_frames[start:end]) -> 여러 시점의 월드 좌표계 pc를 누적
        ref_pose_inv = np.linalg.inv(sample_list[i]["pose"])-> 처리중인 시점의 pose 정보 불러오기
        window_points_local = transform_points_to_local(window_points_world, ref_pose_inv)-> 처리중의 시점으로 누적된 world 좌표계 pc를 변환

        def transform_points_to_local(pts_world, ref_pose_inv):
            if pts_world.shape[0] == 0:
                return pts_world
            N = pts_world.shape[0]
            homo = np.hstack((pts_world, np.ones((N, 1), dtype=np.float32)))
            pts_local = (ref_pose_inv @ homo.T).T[:, :3]
            return pts_local
        """
        detached_feature = [xyz_volume.detach().clone() for xyz_volume in xyz_volumes]
        updated_feature_list = []
        for features_idx in range(len(self.temporal_list[0])):
            updated_feature0 = self.temporal_list[0][features_idx].permute(0,2,3,4,1)
            updated_feature1 = xyz_volumes[features_idx].permute(0,2,3,4,1)
            # updated_feature2 = self.temporal_list[2][features_idx].permute(0,2,3,4,1)
            # updated_feature = self.fusion_conv[features_idx](torch.cat((updated_feature0,updated_feature1,updated_feature2),dim=-1)).permute(0,4,1,2,3)
            updated_feature = self.fusion_conv[features_idx](torch.cat((updated_feature0,updated_feature1),dim=-1)).permute(0,4,1,2,3)
            updated_feature_list.append(updated_feature)
        self.temporal_list.append(detached_feature)
        return updated_feature_list
    
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
        # if len(self.temporal_list) < 1:
        #     self.temporal_list.append([xyz_volume.detach().clone() for xyz_volume in xyz_volumes])
        # elif len(self.temporal_list) == 1:
        #     xyz_volumes = self.temporal_fusion(xyz_volumes, batch_data_samples)
        # else:
        #     self.temporal_list.pop(0)
        #     xyz_volumes = self.temporal_fusion(xyz_volumes, batch_data_samples)
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
    