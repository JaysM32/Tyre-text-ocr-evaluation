tire_data_root = 'data/tiresidewalls/train_mmocr'

tire_rec_train = dict(
    type='OCRDataset',
    data_root=tire_data_root,
    data_prefix=dict(img_path='imgs/'),
    ann_file='labels.json',
    pipeline=None,
    test_mode=False)

tire_rec_test = dict(
    type='OCRDataset',
    data_root=tire_data_root,
    data_prefix=dict(img_path='imgs/'),
    ann_file='labels.json',
    pipeline=None,
    test_mode=True)
