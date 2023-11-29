dataset_type = 'UNITER_VeDataset'

data_root = '/home/datasets/mix_data/UNITER/ve/'

train_datasets = ['train']
test_datasets = ['minival']

ve_cfg = dict(
    train_txt_db=data_root + 've_train.db/',
    train_img_db=data_root + 'flickr30k/',
    val_txt_db=data_root + 've_dev.db/',
    val_img_db=data_root + 'flickr30k/',
    test_txt_db=data_root + 've_test.db/',
    test_img_db=data_root + 'flickr30k/',
    compressed_db=False,
    max_txt_len=60,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=8192,  # 4096
    val_batch_size=8192,  # 4096
)

BUCKET_SIZE = 8192

train_data = dict(
    samples_per_gpu=ve_cfg['train_batch_size'],
    workers_per_gpu=4,
    pin_memory=False,
    batch_sampler=dict(
        type='TokenBucketSampler',
        bucket_size=BUCKET_SIZE,
        batch_size=ve_cfg['train_batch_size'],
        drop_last=True,
        size_multiple=8,
    ),
    data=dict(
        type=dataset_type,
        datacfg=ve_cfg,
        train_or_val=True,
    ),
)

test_data = dict(
    samples_per_gpu=ve_cfg['val_batch_size'],
    workers_per_gpu=4,
    pin_memory=False,
    batch_sampler=dict(
        type='TokenBucketSampler',
        bucket_size=BUCKET_SIZE,
        batch_size=ve_cfg['val_batch_size'],
        drop_last=False,
        size_multiple=8,
    ),
    data=dict(
        type=dataset_type,
        datacfg=ve_cfg,
        train_or_val=False,
    ),
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='UNITER_AccuracyMetric')],
    dataset_converters=[dict(type='UNITER_DatasetConverter')],
)
