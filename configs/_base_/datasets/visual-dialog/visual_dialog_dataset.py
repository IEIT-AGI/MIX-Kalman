dataset_type = 'VisDialDataset'
data_root = '/home/datasets/mix_data/mixk/'
feature_path = 'data/datasets/visdial_data/features/'
annotation_path = 'data/datasets/visdial_data/annotations_npy/'
tokenizer_path = '/home/datasets/mix_data/torch/pytorch_transformers/bert/bert-base-uncased/bert-base-uncased-vocab.txt'

train_datasets = ['train']
test_datasets = ['val']

img_feature_reader = dict(type='ImageFeaturesH5Reader', )

train_reader_cfg = dict(
    type='VisDiaReader',
    mix_features=dict(train=data_root + feature_path + 'visdial_img_feat.lmdb', ),
    mix_annotations=dict(
        train=data_root + annotation_path + 'visdial_1.0_train_processed.npy',
        # train=data_root + annotation_path + 'visdial_1.0_train_processed_small64.npy'
    ),
    image_feature_max_regions=37,
    datasets=train_datasets,  # used datasets
    image_feature_reader=img_feature_reader,
    mask_img_probability=0.15,
)

test_reader_cfg = dict(
    type='VisDiaReader',
    mix_features=dict(val=data_root + feature_path + 'visdial_img_feat.lmdb', ),
    mix_annotations=dict(
        val=data_root + annotation_path + 'visdial_1.0_val_processed.npy',
        # val=data_root + annotation_path + 'visdial_1.0_val_processed_small64.npy',
        dense=data_root + annotation_path + 'visdial_1.0_val_dense_annotations_processed.json'),
    image_feature_max_regions=37,
    datasets=test_datasets,  # used datasets
    image_feature_reader=img_feature_reader,
    mask_img_probability=0,
)

train_info_cpler_cfg = dict(
    type='VisDiaInfoCpler',
    tokenizer=dict(path=tokenizer_path),
    num_options=100,  # number of options to use. [2,100]
    num_negative_samples=1,  # number of negative samples for every positive sample for the nsp loss
    visual_dialog_tot_rounds=11,
    # number of rounds to use in visdial,caption is counted as a separate round, therefore a maximum of 11
    # rounds possible
    max_sequence_len=256,  # maximum sequence length for the dialog sequence
    sequences_per_image=8,  # number of sequences sampled from an image during training
    mask_probability=0.1,  # probability used to sample masked tokens
    has_bert=True,
)

test_info_cpler_cfg = dict(
    type='VisDiaInfoCpler',
    tokenizer=dict(path=tokenizer_path),
    num_options=100,  # number of options to use. [2,100]
    num_negative_samples=1,  # number of negative samples for every positive sample for the nsp loss
    visual_dialog_tot_rounds=11,
    # number of rounds to use in visdial,caption is counted as a separate round, therefore a maximum of 11
    # rounds possible
    max_sequence_len=256,  # maximum sequence length for the dialog sequence
    sequences_per_image=8,  # number of sequences sampled from an image during training
    mask_probability=0,  # probability used to sample masked tokens
    has_bert=True,
)

train_data = dict(
    samples_per_gpu=20,
    workers_per_gpu=2,
    data=dict(type=dataset_type, reader=train_reader_cfg, info_cpler=train_info_cpler_cfg, limit_nums=None),
    drop_last=True,
)

test_data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    data=dict(type=dataset_type, reader=test_reader_cfg, info_cpler=test_info_cpler_cfg, limit_nums=None),
    sampler='DistributedSampler',
    data_last=True,
    is_run_eval=False,
)

post_processor = dict(
    type='Evaluator', metrics=[dict(type='VisDialMetric')], dataset_converters=[dict(type='VisDialDatasetConverter')])
