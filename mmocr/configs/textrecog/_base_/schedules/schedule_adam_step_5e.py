# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr= 0.001 / 8)) # changed to single GPU
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(type='MultiStepLR', milestones=[3, 4], end=5),
]
