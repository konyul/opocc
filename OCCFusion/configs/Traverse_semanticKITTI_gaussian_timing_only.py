_base_ = ['Traverse_semanticKITTI_gaussian.py']

# Just add the custom imports for timing debugging (no progress bar)
custom_imports = dict(
    imports=['occfusion'], 
    allow_failed_imports=False
)

# More frequent logging to see timing info
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),  # Log every 5 iterations
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)