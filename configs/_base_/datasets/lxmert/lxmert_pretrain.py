dataset_type = 'LXMERTPretrainData'
data_root = '/home/datasets/mix_data/'
feature_path = 'lxmert/'
annotation_path = 'lxmert/lxmert/'

train_datasets = ['mscoco_train', 'mscoco_nominival', 'vgnococo']
# train_datasets = ['mscoco_train']
test_datasets = ['mscoco_minival']

train_cfg = dict(
    features=dict(
        mscoco_train=data_root + feature_path + 'mscoco_imgfeat/train2014_obj36.tsv',
        mscoco_nominival=data_root + feature_path + 'mscoco_imgfeat/val2014_obj36.tsv',
        vgnococo=data_root + feature_path + 'vg_gqa_imgfeat/vg_gqa_obj36.tsv',
    ),
    annotations=dict(
        mscoco_train=data_root + annotation_path + 'mscoco_train.json',
        mscoco_nominival=data_root + annotation_path + 'mscoco_nominival.json',
        vgnococo=data_root + annotation_path + 'vgnococo.json',
    ),
    task_matched=True,
    datasets=train_datasets,
)

test_cfg = dict(
    features=dict(mscoco_minival=data_root + feature_path + 'mscoco_imgfeat/val2014_obj36.tsv', ),
    annotations=dict(mscoco_minival=data_root + annotation_path + 'mscoco_minival.json', ),
    datasets=test_datasets,
)

train_data = dict(
    samples_per_gpu=64,  # 16
    workers_per_gpu=0,
    data=dict(type=dataset_type, cfg=train_cfg, limit_nums=2048),
    pin_memory=True,
    drop_last=True,
    collate_fn='lambda x: x')
test_data = dict(samples_per_gpu=8, workers_per_gpu=0, data=dict(type=dataset_type, cfg=test_cfg))

post_processor = dict(
    type='Evaluator', metrics=[dict(type='VQAAccuracyMetric')], dataset_converters=[dict(type='VQADatasetConverter')])
