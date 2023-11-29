dataset_type = 'VQADATASET'
data_root = '/home/datasets/mix_data/mixk/'
feature_path = 'data/datasets/vqa2/grid_features/features/'
annotation_path = 'data/datasets/vqa2/grid_features/annotations/'
feature_default_path = 'data/datasets/vqa2/defaults/features/'
annotation_default_path = 'data/datasets/vqa2/defaults/annotations/'
vocab_path = 'data/datasets/vqa2/defaults/extras/vocabs/'

train_datasets = ['train']
test_datasets = ['oneval']

vqa_reader_train_cfg = dict(
    type='VQAReader',
    card='grid',
    mix_features=dict(
        train=data_root + feature_path + 'train2014',
        val=data_root + feature_path + 'val2014',
        test=data_root + feature_path + 'test2015',
        visualgenome=data_root + feature_path + 'visualgenome',
        oneval=data_root + feature_path + 'val2014'),
    mix_annotations=dict(
        train=data_root + annotation_path + 'imdb_train2014.npy',
        val=data_root + annotation_path + 'imdb_val2014.npy',
        test=data_root + annotation_path + 'imdb_test2015.npy',
        visualgenome=data_root + annotation_path + 'imdb_visualgenome.npy',
        oneval=data_root + annotation_path + 'imdb_oneval2014.npy',
    ),
    datasets=train_datasets  # used datasets
)

vqa_reader_test_cfg = dict(
    type='VQAReader',
    card='grid',
    mix_features=dict(
        train=data_root + feature_path + 'train2014',
        val=data_root + feature_path + 'val2014',
        test=data_root + feature_path + 'test2015',
        visualgenome=data_root + feature_path + 'visualgenome',
        oneval=data_root + feature_path + 'val2014'),
    mix_annotations=dict(
        train=data_root + annotation_path + 'imdb_train2014.npy',
        val=data_root + annotation_path + 'imdb_val2014.npy',
        test=data_root + annotation_path + 'imdb_test2015.npy',
        visualgenome=data_root + annotation_path + 'imdb_visualgenome.npy',
        oneval=data_root + annotation_path + 'imdb_oneval2014.npy',
    ),
    datasets=test_datasets  # used datasets
)

vqa_info_cpler_cfg = dict(
    type='VQAInfoCpler',
    glove_weights=dict(
        glove6b50d=data_root + 'glove.6B.50d.txt.pt',
        glove6b100d=data_root + 'glove.6B.100d.txt.pt',
        glove6b200d=data_root + 'glove.6B.200d.txt.pt',
        glove6b300d=data_root + 'glove.6B.300d.txt.pt',
    ),
    tokenizer='/home/datasets/VQA/bert/' + 'bert-base-uncased-vocab.txt',
    if_bert=False,
    mix_vocab=dict(
        answers_vqa=data_root + vocab_path + 'answers_vqa.txt',
        vocabulart_100k=data_root + vocab_path + 'vocabulary_100k.txt',
        vocabulary_vqa=data_root + vocab_path + 'vocabulary_vqa.txt'))

train_data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    data=dict(type=dataset_type, reader=vqa_reader_train_cfg, info_cpler=vqa_info_cpler_cfg))

test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    data=dict(type=dataset_type, reader=vqa_reader_test_cfg, info_cpler=vqa_info_cpler_cfg))

post_processor = dict(
    type='Evaluator', metrics=[dict(type='VQAAccuracyMetric')], dataset_converters=[dict(type='VQADatasetConverter')])
