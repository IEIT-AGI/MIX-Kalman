dataset_type = 'OSCAR_GQADataset'
data_root = '/home/datasets/mix_data/oscar/datasets/GQA'

train_datasets = ['train']  # 'train+val'
test_datasets = ['val']  # 'test', 'test-dev',

gqa_reader_train_cfg = dict(
    train_data_type='all',
    data_dir=data_root + '/0.4true',
    eval_data_type='bal',
    data_label_type='all',
    task_name='gqa',
    label_file=data_root + '/questions1.2/trainval_testdev_all_ans2label.pkl',
    label2ans_file=data_root + '/questions1.2/trainval_testdev_all_label2ans.pk',
    img_feature_type='faster_r-cnn',
    img_feature_dim=2054,
    img_feature_is_folder=True,
    tokenizer_name=None,
    model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
    load_fast=False,
    model_type='bert',
    max_seq_length=165,
    max_img_seq_length=45,
    output_mode='classification',
    do_lower_case=True,
    name=train_datasets,
    label_pos_feats=None
    # limit_nums=limit_nums,
)

gqa_reader_test_cfg = dict(
    train_data_type='all',
    data_dir=data_root + '/0.4true',
    eval_data_type='bal',
    data_label_type='all',
    task_name='gqa',
    label_file=data_root + '/questions1.2/trainval_testdev_all_ans2label.pkl',
    label2ans_file=data_root + '/questions1.2/trainval_testdev_all_label2ans.pk',
    img_feature_type='faster_r-cnn',
    img_feature_dim=2054,
    img_feature_is_folder=True,
    tokenizer_name=None,
    model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
    load_fast=False,
    model_type='bert',
    max_seq_length=165,
    max_img_seq_length=45,
    output_mode='classification',
    do_lower_case=True,
    name=test_datasets,
    label_pos_feats=None
    # limit_nums=limit_nums,
)

train_data = dict(
    samples_per_gpu=48,
    workers_per_gpu=4,
    data=dict(
        type=dataset_type,
        reader=gqa_reader_train_cfg,
    ),
    sampler='DistributedSampler',
)

test_data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    data=dict(
        type=dataset_type,
        reader=gqa_reader_test_cfg,
    ),
    sampler='SequentialSampler',
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='OSCAR_AccuracyMetric')],
    dataset_converters=[dict(type='OSCAR_DatasetConverter')])
