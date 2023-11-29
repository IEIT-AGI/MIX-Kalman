dataset_type = 'TEXTVQADATASET'
data_root = '/home/datasets/mix_data/mixk/'
feature_path = 'data/datasets/textvqa/defaults/features/open_images/'
ocr_feature_path = 'data/datasets/textvqa/defaults/ocr/'
annotation_path = 'data/datasets/textvqa/defaults/annotations/'
vocab_path = 'data/datasets/textvqa/defaults/extras/vocabs/'

train_datasets = ['train_en']  # train_en
test_datasets = ['val_en']

annotations = dict(
    # train_en=data_root + annotation_path + 'imdb_train_ocr_en_phoc_feature_small_128.npy',
    train_en=data_root + annotation_path + 'imdb_train_ocr_en.npy',
    train_ml=data_root + annotation_path + 'imdb_train_ocr_ml.npy',
    # val_en=data_root + annotation_path + 'imdb_val_ocr_en_phoc_feature_small_128.npy',
    val_en=data_root + annotation_path + 'imdb_val_ocr_en.npy',
    val_ml=data_root + annotation_path + 'imdb_val_ocr_ml.npy',
    test_en=data_root + annotation_path + 'imdb_test_ocr_en.npy',
    test_ml=data_root + annotation_path + 'imdb_test_ocr_ml.npy',
)

textvqa_reader_train_cfg = dict(
    type='TEXTVQAREADER',
    card='default',
    mix_features=dict(
        train_en=data_root + feature_path + 'detectron.lmdb',
        train_ml=data_root + feature_path + 'detectron.lmdb',
        val_en=data_root + feature_path + 'detectron.lmdb',
        val_ml=data_root + feature_path + 'detectron.lmdb',
        test_en=data_root + feature_path + 'detectron.lmdb',
        test_ml=data_root + feature_path + 'detectron.lmdb',
    ),
    mix_global_features=dict(
        train_en=data_root + feature_path + 'resnet152.lmdb',
        train_ml=data_root + feature_path + 'resnet152.lmdb',
        val_en=data_root + feature_path + 'resnet152.lmdb',
        val_ml=data_root + feature_path + 'resnet152.lmdb',
        test_en=data_root + feature_path + 'resnet152.lmdb',
        test_ml=data_root + feature_path + 'resnet152.lmdb',
    ),
    mix_ocr_features=dict(
        train_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        train_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
        val_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        val_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
        test_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        test_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
    ),
    mix_annotations=annotations,
    datasets=train_datasets,
    is_global=True)

textvqa_reader_test_cfg = dict(
    type='TEXTVQAREADER',
    card='default',
    mix_features=dict(
        train_en=data_root + feature_path + 'detectron.lmdb',
        train_ml=data_root + feature_path + 'detectron.lmdb',
        val_en=data_root + feature_path + 'detectron.lmdb',
        val_ml=data_root + feature_path + 'detectron.lmdb',
        test_en=data_root + feature_path + 'detectron.lmdb',
        test_ml=data_root + feature_path + 'detectron.lmdb',
    ),
    mix_global_features=dict(
        train_en=data_root + feature_path + 'resnet152.lmdb',
        train_ml=data_root + feature_path + 'resnet152.lmdb',
        val_en=data_root + feature_path + 'resnet152.lmdb',
        val_ml=data_root + feature_path + 'resnet152.lmdb',
        test_en=data_root + feature_path + 'resnet152.lmdb',
        test_ml=data_root + feature_path + 'resnet152.lmdb',
    ),
    mix_ocr_features=dict(
        train_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        train_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
        val_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        val_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
        test_en=data_root + ocr_feature_path + 'ocr_en/features/ocr_en_frcn_features.lmdb',
        test_ml=data_root + ocr_feature_path + 'ocr_ml/features/ocr_ml_frcn_features.lmdb',
    ),
    mix_annotations=annotations,
    datasets=test_datasets,
    is_global=True)

textvqa_info_cpler_cfg = dict(
    type='TextVQAInfoCpler',
    text_processor=dict(
        type='BertTokenizerProcessor',
        config=dict(
            tokenizer_config=dict(type='bert-base-uncased', params=dict(do_lower_case=True)), max_seq_length=20)),
    answer_processor=dict(
        type='M4CAnswerProcessor',
        config=dict(
            vocab_file=data_root + vocab_path + 'fixed_answer_vocab_textvqa_5k.txt',
            preprocessor=dict(type='simple_word', ),
            context_preprocessor=dict(type='simple_word'),
            max_length=50,
            max_copy_steps=12,
            num_answers=10)),
    copy_processor=dict(type='CopyProcessor', config=dict(max_length=100)),
    phoc_processor=dict(type='PhocProcessor', config=dict(max_length=50)),
    context_processor=dict(type='FastTextProcessor', max_length=50, model_file=data_root + 'fasttext/wiki.en.bin'),
    ocr_token_processor=dict(type='SimpleWordProcessor'),
    bbox_processor=dict(type='BBoxProcessor', max_length=50),
    return_features_info=True,
    use_ocr=True,
    use_ocr_info=True,
    use_order_vectors=True,
    phoc_feature_path=data_root + 'data/datasets/textvqa/defaults/text_vqa_phoc',
)

train_data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    data=dict(type=dataset_type, reader=textvqa_reader_train_cfg, info_cpler=textvqa_info_cpler_cfg, limit_nums=None),
)

test_data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    data=dict(type=dataset_type, reader=textvqa_reader_test_cfg, info_cpler=textvqa_info_cpler_cfg),
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='TextVQAAccuracyMetric')],
    dataset_converters=[
        dict(type='TextVQADatasetConvert', vocab=data_root + vocab_path + 'fixed_answer_vocab_textvqa_5k.txt')
    ])
