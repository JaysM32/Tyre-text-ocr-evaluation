ctw1500_textrecog_data_root = 'data/ctw1500'

ctw1500_textrecog_train = dict(
    type='OCRDataset',
    data_root=ctw1500_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

ctw1500_textrecog_test = dict(
    type='OCRDataset',
    data_root=ctw1500_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
