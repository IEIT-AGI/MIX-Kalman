dataset_type = 'GQADATASET'
data_root = '~/.cache/torch/mixk/'
feature_default_path = 'data/datasets/gqa/defaults/features/'
annotation_default_path = 'data/datasets/gqa/defaults/annotations/'
vocab_path = 'data/datasets/gqa/defaults/extras/vocabs/'

train_datasets = ['train', 'val']
test_datasets = ['val']

vqa_reader_train_cfg = dict(
    type='GQAReader',
    card='default',
    mix_features=dict(
        train=data_root + feature_default_path + 'gqa_train.lmdb',
        val=data_root + feature_default_path + 'gqa_val.lmdb',
        test=data_root + feature_default_path + 'gqa_val.lmdb',
    ),
    mix_annotations=dict(
        train=data_root + annotation_default_path + 'train_balanced_questions.npy',
        val=data_root + annotation_default_path + 'val_balanced_questions.npy',
        test=data_root + annotation_default_path + 'test_balanced_questions.npy',
    ),
    datasets=train_datasets  # used datasets
)

vqa_reader_test_cfg = dict(
    type='GQAReader',
    card='default',
    mix_features=dict(
        train=data_root + feature_default_path + 'gqa_train.lmdb',
        val=data_root + feature_default_path + 'gqa_val.lmdb',
        test=data_root + feature_default_path + 'gqa_val.lmdb',
    ),
    mix_annotations=dict(
        train=data_root + annotation_default_path + 'train_balanced_questions.npy',
        val=data_root + annotation_default_path + 'val_balanced_questions.npy',
        test=data_root + annotation_default_path + 'test_balanced_questions.npy',
    ),
    datasets=test_datasets  # used datasets
)

vqa_info_cpler_cfg = dict(
    type='GQAInfoCpler',
    glove_weights=dict(
        glove6b50d=data_root + 'glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove.6B.300d.txt.pt',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    mix_vocab=dict(
        answers_gqa=data_root + vocab_path + 'answers_gqa.txt',
        vocabulart_100k=data_root + vocab_path + 'vocabulary_100k.txt',
        vocabulary_gqa=data_root + vocab_path + 'vocabulary_gqa.txt'),
    max_seg_lenth=128,  # 20,
    word_mask_ratio=0.0,
    vocab_name='vocabulary_gqa',  # #bert for vocabulary_100k
    vocab_answer_name='answers_gqa',
    glove_name='glove6b300d',
    if_bert=False,
)

train_data = dict(
    samples_per_gpu=127,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=vqa_reader_train_cfg, info_cpler=vqa_info_cpler_cfg),
)

test_data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=vqa_reader_test_cfg, info_cpler=vqa_info_cpler_cfg),
)

post_processor = dict(
    type='Evaluator', metrics=[dict(type='VQAAccuracyMetric')], dataset_converters=[dict(type='VQADatasetConverter')])
