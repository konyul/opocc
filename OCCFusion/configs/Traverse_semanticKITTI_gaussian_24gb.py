_base_ = ['Traverse_semanticKITTI_gaussian.py']

# Optimized for 24GB GPU memory
custom_imports = dict(
    imports=['occfusion', 'occfusion.simple_progress_hook'], 
    allow_failed_imports=False
)

# Increase batch size for better GPU utilization
train_dataloader = dict(
    batch_size=6,  # 2 -> 6 (3x faster training)
    num_workers=6,  # Increase workers for better data loading
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='SemanticKittiSegDataset',
        data_root='/media/spalab/sdb/dhkim/OCCFusion/data/Rellis-3D',
        data_prefix=dict(img_path='',pts_semantic_mask_path='',lidar=''),
        ann_file='/media/spalab/sdb/dhkim/OCCFusion/data_infos/occfusion_train_add_geom.pkl',
        pipeline=[
            dict(
                type='SemanticKITTI_Image_Load',
                to_float32=True,
                color_type='unchanged',
                backend_args=None),
            dict(type='LoadSemanticKITTI_Occupancy'),
            dict(type='LoadSemanticKITTI_Lidar',
                 pc_range=[-25.6, -12.8, -1.6, 0, 12.8, 1.6],
                 with_fov=True),
            dict(
                type='MultiViewWrapper',
                transforms=dict(type='PhotoMetricDistortion3D')),
            dict(
                type='Custom3DPack',
                keys=['img','occ_semantickitti_masked','points'],
                meta_keys=['lidar2img', 'geom_occ_path'])
        ],
        test_mode=False))

val_dataloader = dict(
    batch_size=2,  # 1 -> 2 (faster validation)
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SemanticKittiSegDataset',
        data_root='/media/spalab/sdb/dhkim/OCCFusion/data/Rellis-3D',
        data_prefix=dict(img_path='',pts_semantic_mask_path='',lidar=''),
        ann_file='/media/spalab/sdb/dhkim/OCCFusion/data_infos/occfusion_val_400_900.pkl',
        pipeline=[
            dict(
                type='SemanticKITTI_Image_Load',
                to_float32=False,
                color_type='unchanged',
                backend_args=None),
            dict(type='LoadSemanticKITTI_Occupancy'),
            dict(type='LoadSemanticKITTI_Lidar',
                 pc_range=[-25.6, -12.8, -1.6, 0, 12.8, 1.6]),
            dict(
                type='Custom3DPack',
                keys=['img','occ_semantickitti_masked','points'],
                meta_keys=['lidar2img','sample_id','frame_id', 'geom_occ_path'])
        ],
        test_mode=True))

# Adjust learning rate for larger batch size (linear scaling rule)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, weight_decay=0.01),  # 2e-4 * 3 = 6e-4
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=5000, norm_type=2),
)

# Enhanced logging with progress bar
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),  # More frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'),
    progress_bar=dict(
        type='SimpleProgressBarHook',
        bar_width=100,
        show_eta=True
    )
)

# Enhanced log processor
log_processor = dict(
    type='LogProcessor', 
    window_size=5,
    by_epoch=True
)