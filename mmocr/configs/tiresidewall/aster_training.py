_base_ = [
    '../textrecog/_base_/datasets/totaltext.py',
    '../textrecog/_base_/datasets/cute80.py',
    '../textrecog/_base_/datasets/icdar2015.py',
    '../textrecog/_base_/datasets/ctw1500.py',
    '../textrecog/_base_/default_runtime.py',
    '../textrecog/_base_/schedules/schedule_adamw_cos_6e.py',
    '../textrecog/aster/_base_aster.py',
]

# dataset settings
train_list = [_base_.icdar2015_textrecog_train, _base_.totaltext_text_train]
test_list = [_base_.icdar2015_textrecog_test, _base_.totaltext_textrecog_test]
default_hooks = dict(logger=dict(type='LoggerHook', interval=25))

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))


test_dataloader = val_dataloader

val_evaluator = dict(dataset_prefixes=['IC15','Totaltext'])
test_evaluator = val_evaluator
