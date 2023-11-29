dataset_type = 'OCRVQADATASET'
data_root = '/home/datasets/mix_data/mixk/'
feature_path = 'data/datasets/ocrvqa/defaults/features/'
ocr_feature_path = 'data/datasets/ocrvqa/defaults/ocr_features/'
annotation_path = 'data/datasets/ocrvqa/defaults/annotations/'
vocab_path = 'data/datasets/ocrvqa/defaults/extras/vocabs/'

train_datasets = ['train']
test_datasets = ['val']

reader_train_cfg = dict(
    type='OCRVQAREADER',
    card='default',
    mix_features=dict(
        train=data_root + feature_path + 'detectron.lmdb',
        val=data_root + feature_path + 'detectron.lmdb',
        test=data_root + feature_path + 'detectron.lmdb',
    ),
    mix_ocr_features=dict(
        train=data_root + ocr_feature_path + 'ocr_en_frcn_features.lmdb',
        val=data_root + ocr_feature_path + 'ocr_en_frcn_features.lmdb',
        test=data_root + ocr_feature_path + 'ocr_en_frcn_features.lmdb',
    ),
    mix_annotations=dict(
        train=data_root + annotation_path + 'imdb_train.npy',
        val=data_root + annotation_path + 'imdb_val.npy',
        test=data_root + annotation_path + 'imdb_test.npy',
    ),
    datasets=train_datasets)

reader_test_cfg = dict(
    type='OCRVQAREADER',
    card='default',
    mix_features=dict(
        train=data_root + feature_path + 'detectron.lmdb',
        val=data_root + feature_path + 'detectron.lmdb',
        test=data_root + feature_path + 'detectron.lmdb',
    ),
    mix_ocr_features=dict(
        train=data_root + ocr_feature_path + 'ocr_en_frcn_features.lmdb',
        val=data_root + ocr_feature_path + 'ocr_en_frcn_features.lmdb',
        test=data_root + ocr_feature_path + 'ocr_en_frcn_features.lmdb',
    ),
    mix_annotations=dict(
        train=data_root + annotation_path + 'imdb_train.npy',
        val=data_root + annotation_path + 'imdb_val.npy',
        test=data_root + annotation_path + 'imdb_test.npy',
    ),
    datasets=train_datasets)

info_cpler_cfg = dict(
    type='OCRVQAInfoCpler',
    glove_weights=dict(
        glove6b50d=data_root + 'glove/glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove/glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove/glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove/glove.6B.300d.txt.pt',
    ),
    fasttext_weights=dict(
        wiki300d1m=data_root + 'fasttext/wiki-news-300d-1M.vec',
        wiki300d1msub=data_root + 'fasttext/wiki-news-300d-1M-subword.vec',
        wiki_bin=data_root + 'fasttext/wiki.en.bin',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    mix_vocab=dict(
        answers_st_5k=data_root + vocab_path + 'fixed_answer_vocab_stvqa_5k.txt',
        vocabulary_100k=data_root + vocab_path + 'vocabulary_100k.txt',
    ),
    max_seg_lenth=20,
    max_ocr_lenth=10,
    word_mask_ratio=0.0,
    vocab_name='vocabulary_100k',
    vocab_answer_name='answers_st_5k',
    glove_name='glove6b300d',
    fasttext_name='wiki_bin',
    if_bert=True,
)

train_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=reader_train_cfg, info_cpler=info_cpler_cfg, limit_nums=800))

test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=reader_test_cfg, info_cpler=info_cpler_cfg),
)
