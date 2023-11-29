dataset_type = 'OSCAR_VQADataset'
data_root = '/home/datasets/mix_data/oscar/datasets/vqa'

train_datasets = ['train']  # 'train+val'
test_datasets = ['val']  # 'test2015', 'test-dev2015',

vqa_reader_train_cfg = dict(
    tokenizer_name=None,
    do_lower_case=True,
    model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
    name=train_datasets,
    img_feature_type='faster_r-cnn',
    img_feat_format='pt',
    img_feature_dim=2054,
    img_feature_is_folder=True,
    data_dir=data_root + '/2k',
    label_file=data_root + '/cache/trainval_ans2label.pkl',
    label2ans_file=None,
    txt_data_dir=data_root + '/2k',
    task_name='vqa_text',
    load_fast=False,
    model_type='bert',
    max_seq_length=128,
    output_mode='classification',
    data_label_type='mask',
    use_vg_dev=False,
    use_vg=False,
    max_img_seq_length=50,
    # limit_nums=limit_nums,
)

vqa_reader_test_cfg = dict(
    tokenizer_name=None,
    do_lower_case=True,
    model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
    name=test_datasets,
    img_feature_type='faster_r-cnn',
    img_feat_format='pt',
    img_feature_dim=2054,
    img_feature_is_folder=True,
    data_dir=data_root + '/2k',
    label_file=data_root + '/cache/trainval_ans2label.pkl',
    label2ans_file=None,
    txt_data_dir=data_root + '/2k',
    task_name='vqa_text',
    load_fast=False,
    model_type='bert',
    max_seq_length=128,
    output_mode='classification',
    data_label_type='mask',
    use_vg_dev=False,
    use_vg=False,
    max_img_seq_length=50,
    # limit_nums=limit_nums,
)

train_data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    data=dict(
        type=dataset_type,
        reader=vqa_reader_train_cfg,
    ),
    # sampler='DistributedSampler',
)

test_data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    data=dict(
        type=dataset_type,
        reader=vqa_reader_test_cfg,
    ),
    sampler='SequentialSampler',
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='OSCAR_AccuracyMetric')],
    dataset_converters=[dict(type='OSCAR_DatasetConverter')])
