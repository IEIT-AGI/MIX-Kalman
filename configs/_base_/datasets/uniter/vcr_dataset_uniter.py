dataset_type = 'UNITER_VcrDataset'

dataset_root_dir = '/home/datasets/mix_data/UNITER/vcr/'

train_datasets = ['train']
test_datasets = ['minival']

vcr_cfg = dict(
    train_txt_dbs=[dataset_root_dir + 'vcr_train.db/'],
    train_img_dbs=[
        '{}vcr_gt_train/;{}vcr_train/'.format(dataset_root_dir, dataset_root_dir),
    ],  # two dbs concatenate one string
    val_txt_db=dataset_root_dir + 'vcr_val.db/',
    val_img_db='{}vcr_gt_val/;{}vcr_val/'.format(dataset_root_dir, dataset_root_dir),
    max_txt_len=220,
    conf_th=0.2,
    max_bb=100,
    min_bb=10,
    num_bb=36,
    train_batch_size=16000,  # 4000
    val_batch_size=40,  # 10
)

BUCKET_SIZE = 8192

train_data = dict(
    samples_per_gpu=vcr_cfg['train_batch_size'],
    workers_per_gpu=0,
    pin_memory=True,
    batch_sampler=dict(
        type='TokenBucketSampler',
        bucket_size=BUCKET_SIZE,
        batch_size=vcr_cfg['train_batch_size'],
        drop_last=True,
        size_multiple=8,
    ),
    data=dict(
        type=dataset_type,
        datacfg=vcr_cfg,
        train_or_val=True,
    ),
)

test_data = dict(
    samples_per_gpu=vcr_cfg['val_batch_size'],
    workers_per_gpu=0,
    pin_memory=True,
    data=dict(
        type=dataset_type,
        datacfg=vcr_cfg,
        train_or_val=False,
    ),
)

post_processor = dict(
    type='Evaluator',
    metrics=[
        dict(
            type='UNITER_VCR_AccuracyMetric',
            name='vcr q->a',
            metric_key='qa_batch_score',
        ),
        dict(
            type='UNITER_VCR_AccuracyMetric',
            name='vcr qa->r',
            metric_key='qar_batch_score',
        ),
        dict(
            type='UNITER_VCR_AccuracyMetric',
            name='vcr q->ar',
            metric_key='batch_score',
        ),
    ],
    dataset_converters=[dict(type='UNITER_VCR_DatasetConverter')],
)
