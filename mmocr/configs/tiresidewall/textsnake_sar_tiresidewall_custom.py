_base_ = [
    '../textrecog/sar/_base_sar_resnet31_parallel-decoder.py',
    '../textrecog/_base_/schedules/schedule_adam_step_5e.py',
    '../textrecog/_base_/default_runtime.py',
]

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

dataset_type = 'OCRDataset'
root = '/mmocr/data/tiresidewalls/train' # Location where the annotation and crop images are being stored

img_prefix ='/mmocr/data/tiresidewalls/train'
train_anno_file1 = '/mmocr/data/tiresidewalls/_annotations.coco.jsonl'


train_dataloader = dict(type='AnnFileLoader',
                            repeat=100,                   
                            file_format='txt',  # only txt and lmdb
                            file_storage_backend='disk',
                            parser=dict(type='LineJsonParser',
                                        keys=['filename', 'text']))

train_datasets1 = dict(type='OCRDataset',
                       img_prefix=img_prefix,
                       ann_file=train_anno_file1,
                       loader=train_dataloader,
                       pipeline=None,           
                       test_mode=False)


train_list = [train_datasets1]


val_dataloader = dict(type='AnnFileLoader',
                            repeat=1,                   
                            file_format='txt',  # only txt and lmdb
                            file_storage_backend='disk',
                            parser=dict(type='LineJsonParser',
                                        keys=['filename', 'text']))

train_datasets1 = dict(type='OCRDataset',
                       img_prefix=img_prefix,
                       ann_file=train_anno_file1,
                       loader=val_dataloader,
                       pipeline=None,           
                       test_mode=False)

test_dataloader = val_dataloader

test_list =  [train_datasets1]



work_dir = '/mmocr/demo/tutorial_exps'


data = dict(
    workers_per_gpu=2,
    samples_per_gpu=8,
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline)
    )

evaluation = dict(interval=1, metric='acc')
