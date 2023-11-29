dataset_type = 'OSCAR_NLVR2Dataset'
data_root = '/home/datasets/mix_data/oscar/datasets/nlvr2'

train_datasets = ['train']
test_datasets = ['val']  # 'test1'

nlvr_reader_train_cfg = dict(
    data_dir=data_root + '/ft_corpus',
    eval_data_type='all',
    use_label_seq=False,
    data_label_type='all',
    img_feature_type='faster_r-cnn',
    img_feature_dim=2054,
    img_feature_is_folder=True,
    task_name='nlvr',
    max_seq_length=55,
    output_mode='classification',
    use_pair=True,
    model_type='bert',
    max_img_seq_length=40,
    tokenizer_name=None,
    do_lower_case=True,
    model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
    name=train_datasets,
    # limit_nums=limit_nums,
)

nlvr_reader_test_cfg = dict(
    data_dir=data_root + '/ft_corpus',
    eval_data_type='all',
    use_label_seq=False,
    data_label_type='all',
    img_feature_type='faster_r-cnn',
    img_feature_dim=2054,
    img_feature_is_folder=True,
    task_name='nlvr',
    max_seq_length=55,
    output_mode='classification',
    use_pair=True,
    model_type='bert',
    max_img_seq_length=40,
    tokenizer_name=None,
    do_lower_case=True,
    model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
    name=test_datasets,
    # limit_nums=limit_nums,
)

train_data = dict(
    samples_per_gpu=72,
    workers_per_gpu=4,
    data=dict(
        type=dataset_type,
        reader=nlvr_reader_train_cfg,
    ),
    sampler='RandomSampler',
)

test_data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    data=dict(
        type=dataset_type,
        reader=nlvr_reader_test_cfg,
    ),
    sampler='SequentialSampler',
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='OSCAR_AccuracyMetric')],
    dataset_converters=[dict(type='OSCAR_DatasetConverter')])
