dataset_type = 'VCRDATASET'
data_root = '/home/datasets/VCR/'
feature_default_path = ''
annotation_default_path = data_root
# vocab_path = 'data/datasets/gqa/defaults/extras/vocabs/'

train_datasets = ['train']
test_datasets = ['val']
mode = 'answer'
default_answer_idx = 0

vcr_reader_train_cfg = dict(
    type='VCRReader',
    card='default',
    mode=mode,
    image_dir=data_root + 'vcr1images/',
    annotations=dict(
        train=data_root + 'train.jsonl',
        val=data_root + 'val.jsonl',
        test=data_root + 'test.jsonl',
    ),
    text_infos=dict(
        answer=dict(
            train=data_root + 'bert_da_answer_train.h5',
            val=data_root + 'bert_da_answer_val.h5',
            test=data_root + 'bert_da_answer_test.h5',
        ),
        rationale=dict(
            train=data_root + 'bert_da_rationale_train.h5',
            val=data_root + 'bert_da_rationale_val.h5',
            test=data_root + 'bert_da_rationale_test.h5',
        ),
    ),
    coco_cate_dir=data_root + 'cocoontology.json',
    default_answer_idx=default_answer_idx,
    datasets=train_datasets,  # used datasets
    is_train=True,
)

vcr_reader_test_cfg = dict(
    type='VCRReader',
    card='default',
    mode=mode,
    image_dir=data_root + 'vcr1images/',
    annotations=dict(
        train=data_root + 'train.jsonl',
        val=data_root + 'val.jsonl',
        test=data_root + 'test.jsonl',
    ),
    text_infos=dict(
        answer=dict(
            train=data_root + 'bert_da_answer_train.h5',
            val=data_root + 'bert_da_answer_val.h5',
            test=data_root + 'bert_da_answer_test.h5',
        ),
        rationale=dict(
            train=data_root + 'bert_da_rationale_train.h5',
            val=data_root + 'bert_da_rationale_val.h5',
            test=data_root + 'bert_da_rationale_test.h5',
        ),
    ),
    coco_cate_dir=data_root + 'cocoontology.json',
    default_answer_idx=default_answer_idx,
    datasets=test_datasets,  # used datasets
    is_train=False,
)

vcr_info_cpler_cfg = dict(
    type='VCRInfoCpler',
    default_max_length=20,
    max_ques_length=20,
    max_answer_length=20,
    max_obj_length=50,
)

train_data = dict(
    samples_per_gpu=4,  # 16
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=vcr_reader_train_cfg, info_cpler=vcr_info_cpler_cfg, limit_nums=None))

test_data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    data=dict(type=dataset_type, reader=vcr_reader_test_cfg, info_cpler=vcr_info_cpler_cfg),
)

post_processor = dict(
    type='Evaluator', metrics=[dict(type='VCRAccuracyMetric')], dataset_converters=[dict(type='VCRDatasetConverter')])
