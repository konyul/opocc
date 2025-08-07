# Simple configuration for Rellis3D with 3 classes
_base_ = []

# Plugin
plugin = True
plugin_dir = "projects/occ_plugin/"

# Dataset
dataset_type = 'OCCFusionWrapper'
data_root = '/mnt/sdb/kypark/OCCFusion/data/Rellis-3D/'
train_ann_file = '/mnt/sdb/kypark/OCCFusion/data_infos/occfusion_train_add_geom.pkl'
val_ann_file = '/mnt/sdb/kypark/OCCFusion/data_infos/occfusion_val_final_0_2000.pkl'

# Classes - Rellis3D has 3 classes
class_names = ['empty', 'traversable', 'non-traversable']
num_classes = 3
empty_idx = 0

# Grid settings from GaussianFormer
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

# Model
model = dict(
    type='OccNet',
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE', 
        num_features=4),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=16,
        out_channel=256,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        sparse_shape_xyz=occ_size),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=256,
        block_inplanes=[64, 128, 256, 512],
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01)),
    occ_encoder_neck=dict(
        type='FPN3D',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01)),
    pts_bbox_head=dict(
        type='OccHeadRellis',
        in_channels=256,
        out_channel=3,  # 3 classes for Rellis3D
        num_level=1,
        soft_weights=False,
        cascade_ratio=1,
        sample_from_voxel=False,
        sample_from_img=False,
        final_occ_size=occ_size,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0),
        empty_idx=empty_idx,
        visible_loss=False,
        balance_cls_weight=True),
    empty_idx=empty_idx)

# Data pipeline
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

# Dataset
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
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

# Training
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
workflow = [('train', 1)]

# Evaluation
evaluation = dict(interval=1, pipeline=test_pipeline)

# Logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
custom_hooks = []

# Runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(1)
seed = 0
deterministic = False