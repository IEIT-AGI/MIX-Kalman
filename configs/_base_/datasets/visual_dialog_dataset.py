dataset_type = 'VisDialDataset'
data_root = '/home/datasets/mix_data/mixk/'
feature_path = 'data/datasets/visdial_data/features/'
annotation_path = 'data/datasets/visdial_data/annotations_npy/'

# train_datasets = ["train", "val", "visualgenome"]
# train_datasets = ["train"]
# test_datasets = ["oneval"]
# test_datasets = ["test"]

train_datasets = ['train']
test_datasets = ['val']

img_feature_reader = dict(type='ImageFeaturesH5Reader', )

vqa_reader_train_cfg = dict(
    type='VisDiaReader',
    mix_features=dict(train=data_root + feature_path + 'visdial_img_feat.lmdb', ),
    mix_annotations=dict(train=data_root + annotation_path + 'visdial_1.0_train_processed.npy', ),
    image_feature_max_regions=37,
    datasets=train_datasets,  # used datasets
    image_feature_reader=img_feature_reader,
)

vqa_reader_test_cfg = dict(
    type='VisDiaReader',
    mix_features=dict(val=data_root + feature_path + 'visdial_img_feat.lmdb', ),
    mix_annotations=dict(
        val=data_root + annotation_path + 'visdial_1.0_val_processed.npy',
        dense=data_root + annotation_path + 'visdial_1.0_val_dense_annotations_processed.json'),
    image_feature_max_regions=37,
    datasets=test_datasets,  # used datasets
    image_feature_reader=img_feature_reader,
)

visual_dialog_info_cpler_cfg = dict(
    type='VisDiaInfoCpler',
    tokenizer=dict(path='', ),
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

train_data = dict(
    samples_per_gpu=4,  # 16
    workers_per_gpu=2,
    data=dict(type=dataset_type, reader=vqa_reader_train_cfg, info_cpler=visual_dialog_info_cpler_cfg, limit_nums=None))

test_data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    data=dict(type=dataset_type, reader=vqa_reader_test_cfg, info_cpler=visual_dialog_info_cpler_cfg, limit_nums=None),
)

post_processor = dict(
    type='Evaluator', metrics=[dict(type='VisDialMetric')], dataset_converters=[dict(type='VisDialDatasetConverter')])
