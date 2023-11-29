dataset_type = 'GQATorchDataset'
data_root = '/home/datasets/mix_data/lxmert/'
feature_path = 'vg_gqa_imgfeat/'
annotation_path = 'gqa/'

train_datasets = ['train', 'valid']
test_datasets = ['testdev']

gqa_reader_train_cfg = dict(
    annotations=dict(
        train=data_root + annotation_path + 'train.json',
        valid=data_root + annotation_path + 'valid.json',
        testdev=data_root + annotation_path + 'testdev.json',
    ),
    answer_2_label=data_root + annotation_path + 'trainval_ans2label.json',
    label_2_answer=data_root + annotation_path + 'trainval_label2ans.json',
    datasets=train_datasets,  # used datasets
    # topk=512,
    img_feature=dict(
        train=data_root + feature_path + 'vg_gqa_obj36.tsv',
        testdev=data_root + feature_path + 'gqa_testdev_obj36.tsv',
    ))

gqa_reader_test_cfg = dict(
    annotations=dict(
        train=data_root + annotation_path + 'train.json',
        valid=data_root + annotation_path + 'valid.json',
        testdev=data_root + annotation_path + 'testdev.json',
    ),
    answer_2_label=data_root + annotation_path + 'trainval_ans2label.json',
    label_2_answer=data_root + annotation_path + 'trainval_label2ans.json',
    # topk=512,
    datasets=test_datasets,  # used datasets
    img_feature=dict(
        train=data_root + feature_path + 'vg_gqa_obj36.tsv',
        testdev=data_root + feature_path + 'gqa_testdev_obj36.tsv',
    ))

train_data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    data=dict(
        type=dataset_type,
        reader=gqa_reader_train_cfg,
    ),
    drop_last=True,
    shuffle=True,
)

test_data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=gqa_reader_test_cfg),
    drop_last=False,
    shuffle=False,
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='LXMERT_VQAAccuracyMetric', cfg=gqa_reader_test_cfg, task='GQA')],
    dataset_converters=[dict(type='LXMERT_VQADatasetConverter')])
