_base_ = [
    # '../_base_/default_runtime.py'  # Skip base config for now
]

# Plugin
plugin = True
plugin_dir = "projects/occ_plugin/"

# Dataset settings
dataset_type = 'OCCFusionWrapper'
data_root = '/mnt/sdb/kypark/OCCFusion/data/Rellis-3D/'
train_ann_file = '/mnt/sdb/kypark/OCCFusion/data_infos/occfusion_train_add_geom.pkl'
val_ann_file = '/mnt/sdb/kypark/OCCFusion/data_infos/occfusion_val_final_0_2000.pkl'

# Class settings
class_names = ['empty', 'traverse', 'non_traverse']
num_classes = 3
empty_idx = 0

# Occupancy settings
point_cloud_range = [-25.6, -12.8, -1.6, 0.0, 12.8, 1.6]
occ_size = [256, 256, 32]
voxel_size = [0.1, 0.1, 0.1]

# Input modality
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

# Model - Minimal configuration with regular BN
model = dict(
    type='OccNet',
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=16,
        out_channel=256,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        sparse_shape_xyz=[256, 256, 16]),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=256,
        block_inplanes=[80, 160, 320, 640],
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01)),
    occ_encoder_neck=dict(
        type='FPN3D',
        in_channels=[80, 160, 320, 640],
        out_channels=256,
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01)),
    pts_bbox_head=dict(
        type='OccHead',  # Use original OccHead
        in_channels=[256, 256, 256, 256],
        out_channel=num_classes,
        num_level=4,
        soft_weights=False,
        cascade_ratio=1,
        final_occ_size=occ_size,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=0.0,
            loss_voxel_geo_scal_weight=0.0,
            loss_voxel_lovasz_weight=0.0),
        empty_idx=empty_idx,
        balance_cls_weight=False),  # Disable for simplicity
    empty_idx=empty_idx)

# Minimal pipeline
train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4),
    dict(type='LoadOccupancyRellis',
         to_float32=True,
         use_semantic=True,
         occ_path=data_root,
         grid_size=occ_size,
         unoccupied=empty_idx,
         pc_range=point_cloud_range),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_occ'])
]

test_pipeline = train_pipeline.copy()

# Dataset config
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

# Simple optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)

# Learning rate - constant
lr_config = dict(policy='fixed')

# Runtime
runner = dict(type='EpochBasedRunner', max_epochs=1)  # Just 1 epoch for testing
workflow = [('train', 1)]

# Minimal evaluation config
evaluation = dict(interval=999999, pipeline=test_pipeline)  # Very large interval to skip evaluation

# Logging
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook')])

# No custom hooks
custom_hooks = []

# Checkpoint
checkpoint_config = dict(interval=1)

# GPU
gpu_ids = [0]

# Random seed
seed = 0
deterministic = True

# Distributed parameters
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None