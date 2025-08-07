_base_ = [
    '../datasets/custom_rellis-3d.py',
    '../_base_/default_runtime.py'
]

# Plugin configuration
plugin = True
plugin_dir = "projects/occ_plugin/"

# Data configuration - LiDAR only
input_modality = dict(
    use_lidar=True,
    use_camera=False,  # Disable camera for LiDAR-only
    use_radar=False,
    use_map=False,
    use_external=False)

# Dataset paths
data_root = '/mnt/sdb/kypark/OCCFusion/data/Rellis-3D/'
occ_path = data_root
train_ann_file = "/mnt/sdb/kypark/OCCFusion/data_infos/occfusion_train_add_geom.pkl"
val_ann_file = "/mnt/sdb/kypark/OCCFusion/data_infos/occfusion_val_final_0_2000.pkl"

# Class names for Rellis-3D (3 classes)
class_names = ['empty', 'traverse', 'non_traverse']
num_cls = 3
empty_idx = 0

# Point cloud range and occupancy grid config
point_cloud_range = [-25.6, -12.8, -1.6, 0.0, 12.8, 1.6]
occ_size = [256, 256, 32]
voxel_channels = [80, 160, 320, 640]
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

# Voxel size
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

# Model parameters
cascade_ratio = 1
sample_from_voxel = False
numC_Trans = 80

# Dataset type
dataset_type = 'rellisOCCDataset'

# Model - LiDAR only
model = dict(
    type='OccNet',
    loss_norm=True,
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
        out_channel=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[256, 256, 16],
    ),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=256,
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
        final_occ_size=occ_size,
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
        class_weight=[0.01, 5, 1],
        balance_cls_weight=True,
    ),
    empty_idx=empty_idx,
)

# Training pipeline - LiDAR only
train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4),
    dict(type='LoadOccupancyRellis',
         to_float32=True,
         use_semantic=True,
         occ_path=occ_path,
         grid_size=occ_size,
         use_vel=False,
         unoccupied=empty_idx,
         pc_range=point_cloud_range,
         cal_visible=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_occ']),
]

# Test pipeline - LiDAR only
test_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4),
    dict(type='LoadOccupancyRellis',
         to_float32=True,
         use_semantic=True,
         occ_path=occ_path,
         grid_size=occ_size,
         use_vel=False,
         unoccupied=empty_idx,
         pc_range=point_cloud_range,
         cal_visible=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]

# Dataset configuration
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        test_mode=False,
        box_type_3d='LiDAR'
    ),
    val=dict(
        type=dataset_type,
        occ_root=occ_path,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
    test=dict(
        type=dataset_type,
        occ_root=occ_path,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate config
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# Training schedule
runner = dict(type='EpochBasedRunner', max_epochs=24)

# Evaluation config
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

# Custom hooks
custom_hooks = [
    dict(type='OccEfficiencyHook'),
]

# Load from
load_from = None