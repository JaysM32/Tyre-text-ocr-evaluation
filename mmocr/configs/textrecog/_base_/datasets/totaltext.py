totaltext_data_root = 'data/totaltext'

totaltext_text_train = dict(
    type='OCRDataset',
    data_root=totaltext_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

totaltext_textrecog_test = dict(
    type='OCRDataset',
    data_root=totaltext_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
