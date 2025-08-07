# Use Rellis-3D dataset and runtime defaults
_base_ = [
    '../datasets/custom_rellis-3d.py',
    '../_base_/default_runtime.py'
]

# Input modality
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

plugin = True
plugin_dir = "projects/occ_plugin/"

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# Dataset paths
data_root = 'data/rellis-3d/'
occ_path = data_root + 'occupancy'
depth_gt_path = data_root + 'depth_gt'
train_ann_file = "./data_infos/occfusion_train_baseline.pkl"
val_ann_file = "./data_infos/occfusion_val_final_0_2000.pkl"

# Rellis-3D classes (empty + traversable + non-traversable)
class_names = ['traverse', 'non_traverse']
num_cls = 3  # includes empty
empty_idx = 0

# Point cloud / voxel settings
point_cloud_range = [-25.6, -12.8, -1.6, 0.0, 12.8, 1.6]
occ_size = [256, 256, 32]
lss_downsample = [4, 4, 4]
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_channels = [80, 160, 320, 640]
visible_mask = False

cascade_ratio = 4
sample_from_voxel = False
sample_from_img = False

dataset_type = 'rellisOCCDataset'
file_client_args = dict(backend='disk')

# Rellis-3D uses a single front camera
data_config = {
    'cams': ['CAM_FRONT'],
    'Ncams': 1,
    'input_size': (864, 1600),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# downsample ratio in [x, y, z] when generating 3D volumes in LSS
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x*lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y*lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z*lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}
numC_Trans = 80
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

model = dict(
    type='OccNet',
    loss_norm=True,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(type='ViewTransformerLiftSplatShootVoxel',
                              norm_cfg=dict(type='SyncBN', requires_grad=True),
                              loss_depth_weight=3.,
                              loss_depth_type='kld',
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=numC_Trans,
                              vp_megvii=False),
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=[0.1, 0.1, 0.1],
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=16,
        out_channel=numC_Trans,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[256, 256, 16],
    ),
    occ_fuser=dict(
        type='ConvFuser',
        in_channels=numC_Trans,
        out_channels=numC_Trans,
    ),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    pts_bbox_head=dict(
        type='OccHeadRellis',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=10.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=0.0,
        ),
        class_weight=[0.01, 5.0, 1.0],
        balance_cls_weight=True,
    ),
    empty_idx=empty_idx,
)

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4),
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config,
         sequential=False, aligned=True, trans_only=False, depth_gt_path=depth_gt_path,
         mmlabnorm=True, load_depth=True, img_norm_cfg=img_norm_cfg),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         input_modality=input_modality),
    dict(type='LoadOccupancyRellis', to_float32=True, use_semantic=True, occ_path=occ_path,
         grid_size=occ_size, use_vel=False, unoccupied=empty_idx,
         pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4),
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config, depth_gt_path=depth_gt_path,
         sequential=False, aligned=True, trans_only=False, mmlabnorm=True, img_norm_cfg=img_norm_cfg),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         input_modality=input_modality,
         is_train=False),
    dict(type='LoadOccupancyRellis', to_float32=True, use_semantic=True, occ_path=occ_path,
         grid_size=occ_size, use_vel=False, unoccupied=empty_idx,
         pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points'],
         meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]


test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=True
)
train_config=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        occ_size=occ_size,
        pc_range=point_cloud_range),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]

