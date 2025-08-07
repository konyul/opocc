_base_ = ['Traverse_semanticKITTI_gaussian.py']

# Add progress bar and timing to existing config
custom_imports = dict(
    imports=['occfusion', 'occfusion.simple_progress_hook'], 
    allow_failed_imports=False
)

# Enhanced logging with progress bar
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),  # More frequent logging for better progress tracking
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'),
    # Add our simple progress bar hook (more stable)
    progress_bar=dict(
        type='SimpleProgressBarHook',
        bar_width=100,
        show_eta=True
    )
)

# Enhanced log processor for better output
log_processor = dict(
    type='LogProcessor', 
    window_size=10,  # Smaller window for more responsive updates
    by_epoch=True
)