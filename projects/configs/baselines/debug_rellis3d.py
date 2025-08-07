# Debug configuration for Rellis3D
_base_ = []

# Plugin
plugin = True
plugin_dir = "projects/occ_plugin/"

# Dataset
dataset_type = 'OCCFusionWrapper'
data_root = '/mnt/sdb/kypark/OCCFusion/data/Rellis-3D/'
train_ann_file = '/mnt/sdb/kypark/OCCFusion/data_infos/occfusion_train_add_geom.pkl'
val_ann_file = '/mnt/sdb/kypark/OCCFusion/data_infos/occfusion_val_final_0_2000.pkl'

# Classes
class_names = ['empty', 'traverse', 'non_traverse']
num_classes = 3
empty_idx = 0

# Grid settings - match model expectations
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]  # Simplified range
occ_size = [200, 200, 16]  # Simplified size
voxel_size = [0.4, 0.4, 0.4]  # (80/200, 80/200, 6.4/16)

# Input modality
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

# Very simple model
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
        sparse_shape_xyz=occ_size),  # Use occ_size directly
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
        cascade_ratio=1,  # No cascade
        sample_from_voxel=False,
        sample_from_img=False,
        final_occ_size=occ_size,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=0.0,
            loss_voxel_geo_scal_weight=0.0,
            loss_voxel_lovasz_weight=0.0),
        empty_idx=empty_idx,
        visible_loss=False,
        balance_cls_weight=True),
    empty_idx=empty_idx)

# Override LoadOccupancy to resize gt_occ
train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=4,
         use_dim=4),
    dict(type='LoadOccupancyRellis',
         to_float32=True,
         use_semantic=True,
         occ_path=data_root,
         grid_size=[200, 200, 16],  # Match model expectation
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

# Training
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='fixed')

runner = dict(type='EpochBasedRunner', max_epochs=1)
workflow = [('train', 1)]

evaluation = dict(interval=999999, pipeline=test_pipeline)

# Logging
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook')])

checkpoint_config = dict(interval=1)
custom_hooks = []

# Runtime
gpu_ids = [0]
seed = 0
deterministic = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None