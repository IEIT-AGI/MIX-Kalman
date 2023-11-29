dataset_type = 'UNITER_NLVR2Dataset'

dataset_root_dir = '/home/datasets/mix_data/UNITER/nlvr2/'

train_datasets = ['train']
test_datasets = ['minival']  # name not in use, but have defined one to run

nlvr_cfg = dict(
    train_txt_db=dataset_root_dir + 'nlvr2_train.db',
    train_img_db=dataset_root_dir + 'nlvr2_train/',
    val_txt_db=dataset_root_dir + 'nlvr2_dev.db',
    val_img_db=dataset_root_dir + 'nlvr2_dev/',
    test_txt_db=dataset_root_dir + 'nlvr2_test1.db',
    test_img_db=dataset_root_dir + 'nlvr2_test/',
    use_img_type=True,
    max_txt_len=60,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=10240,
    val_batch_size=10240,
)

BUCKET_SIZE = 8192

train_data = dict(
    samples_per_gpu=nlvr_cfg['train_batch_size'],
    workers_per_gpu=0,
    pin_memory=True,
    batch_sampler=dict(
        type='TokenBucketSampler',
        bucket_size=BUCKET_SIZE,
        batch_size=nlvr_cfg['train_batch_size'],
        drop_last=True,
        size_multiple=8,
    ),
    data=dict(
        type=dataset_type,
        datacfg=nlvr_cfg,
        train_or_val=True,
    ),
)

test_data = dict(
    samples_per_gpu=nlvr_cfg['val_batch_size'],
    workers_per_gpu=0,
    batch_sampler=dict(
        type='TokenBucketSampler',
        bucket_size=BUCKET_SIZE,
        batch_size=nlvr_cfg['val_batch_size'],
        drop_last=False,
        size_multiple=8,
    ),
    pin_memory=True,
    data=dict(
        type=dataset_type,
        datacfg=nlvr_cfg,
        train_or_val=False,
    ),
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='UNITER_AccuracyMetric')],
    dataset_converters=[dict(type='UNITER_DatasetConverter')],
)
