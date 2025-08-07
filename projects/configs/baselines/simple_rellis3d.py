_base_ = [
    '../_base_/default_runtime.py'
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

# Model - Very simple occupancy head for testing
model = dict(
    type='OccNet',
    # Skip image backbone since we're testing data loading first
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
        norm_cfg=dict(type='BN', requires_grad=True),
        sparse_shape_xyz=[256, 256, 16]),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=256,
        block_inplanes=[80, 160, 320, 640],
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True)),
    occ_encoder_neck=dict(
        type='FPN3D',
        in_channels=[80, 160, 320, 640],
        out_channels=256,
        norm_cfg=dict(type='BN', requires_grad=True)),
    pts_bbox_head=dict(
        type='OccHeadRellis',
        in_channels=[256, 256, 256, 256],
        out_channel=num_classes,
        num_level=4,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_lovasz_weight=0.0),
        empty_idx=empty_idx,
        balance_cls_weight=True,
        class_weight=[0.01, 5.0, 1.0]),
    empty_idx=empty_idx)

# Simple pipeline - just load point cloud and occupancy
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_occ'])
]

test_pipeline = [
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
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_occ'])
]

# Dataset config
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,  # Set to 0 for debugging
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

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# Runtime
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1)
workflow = [('train', 1)]  # train 1 epoch

# Logging
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')])

# Remove OccEfficiencyHook that might cause issues
custom_hooks = []

# GPU config
gpu_ids = [0]