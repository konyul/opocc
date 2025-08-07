_base_ = ['OccFusion.py']

# Add progress bar hook
custom_imports = dict(
    imports=['occfusion', 'occfusion.progress_hook'], 
    allow_failed_imports=False
)

# Enhanced logging with progress bar
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),  # More frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'),
    # Add our custom progress bar hook
    progress_bar=dict(
        type='ProgressBarHook',
        bar_width=100,
        show_eta=True,
        show_loss=True,
        update_interval=1
    )
)

# Enhanced log processor for better output
log_processor = dict(
    type='LogProcessor', 
    window_size=20,  # Smaller window for more responsive loss updates
    by_epoch=True,
    custom_cfg=[
        # Track GPU memory usage
        dict(log_name='memory', window_size=1),
        # Track timing information
        dict(log_name='time', window_size=10),
        # Track data loading time
        dict(log_name='data_time', window_size=10),
    ]
)

# Reduce checkpoint saving to every 2 epochs to speed up training
default_hooks['checkpoint'] = dict(type='CheckpointHook', interval=2, max_keep_ckpts=3)

# Enable more detailed logging
log_level = 'INFO'

# Optional: Enable TensorBoard logging for additional monitoring
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend', save_dir='./work_dirs/tb_logs'),
]

visualizer = dict(
    type='Det3DLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)