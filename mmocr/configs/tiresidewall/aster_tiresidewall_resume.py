_base_ = [
    '../textrecog/_base_/datasets/tire_data.py',
    '../textrecog/_base_/default_runtime.py',
    '../textrecog/_base_/schedules/schedule_adam_step_5e.py',
    '../textrecog/aster/_base_aster.py',
]

# dataset settings
train_list = [_base_.tire_rec_train]
test_list = [_base_.tire_rec_test]
default_hooks = dict(logger=dict(type='LoggerHook', interval=10))

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))


test_dataloader = val_dataloader

val_evaluator = dict(dataset_prefixes=['Tire'])
test_evaluator = val_evaluator
