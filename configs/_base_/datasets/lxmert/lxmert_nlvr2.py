dataset_type = 'NLVR2TorchDataset'
data_root = '/home/datasets/mix_data/lxmert/'
feature_path = 'nlvr2_imgfeat/'
annotation_path = 'nlvr2/'

train_datasets = ['train']
test_datasets = ['valid']

nlvr_reader_train_cfg = dict(
    annotations=dict(
        train=data_root + annotation_path + 'train.json',
        valid=data_root + annotation_path + 'valid.json',
    ),
    datasets=train_datasets,  # used datasets

    # topk=512,
    img_feature=dict(
        train=data_root + feature_path + 'train_obj36.tsv',
        valid=data_root + feature_path + 'valid_obj36.tsv',
    ))

nlvr_reader_test_cfg = dict(
    annotations=dict(
        train=data_root + annotation_path + 'train.json',
        valid=data_root + annotation_path + 'valid.json',
    ),
    # topk=512,
    datasets=test_datasets,  # used datasets
    img_feature=dict(
        train=data_root + feature_path + 'train_obj36.tsv',
        valid=data_root + feature_path + 'valid_obj36.tsv',
    ))

train_data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    data=dict(
        type=dataset_type,
        reader=nlvr_reader_train_cfg,
        # limit_nums=400,
    ),
    drop_last=True,
    shuffle=True,
)

test_data = dict(
    samples_per_gpu=512,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=nlvr_reader_test_cfg),
    drop_last=False,
    shuffle=False,
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='LXMERT_VQAAccuracyMetric', cfg=nlvr_reader_test_cfg, task='NLVR')],
    dataset_converters=[dict(type='LXMERT_VQADatasetConverter')])
