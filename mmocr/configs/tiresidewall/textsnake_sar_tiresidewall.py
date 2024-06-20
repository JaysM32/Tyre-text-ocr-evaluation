# The new config inherits a base config to highlight the necessary modification (four predefine configurations of the model, data $ pipeline, training schedule and runtime setting as follow.)
_base_ = "../textrecog/sar/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real.py"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox=dict(num_classes=36))

# Modify dataset related settings
data_root = 'data/tiresidewalls/'
metainfo = {
    'classes': ('0','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'),
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotation.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotation.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotation.coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmocr/textrecog/sar/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real_20220915_185451-1fd6b1fc.pth'
