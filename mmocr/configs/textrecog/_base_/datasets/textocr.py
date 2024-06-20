textocr_textrecog_data_root = 'data/textocr'

textocr_textrecog_train = dict(
    type='OCRDataset',
    data_root=textocr_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

textocr_textrecog_test = dict(
    type='OCRDataset',
    data_root=textocr_textrecog_data_root,
    ann_file='textrecog_test.json',
    test_mode=True,
    pipeline=None)
