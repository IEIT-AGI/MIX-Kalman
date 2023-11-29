from configs._base_.datasets.vilbert.vilbert_task_config import (
    task_ids,
    TASKS,
)

dataset_type = 'LoadDatasets'

# test mode directly read this data set
test_datasets = [TASKS['TASK' + task_ids]['val_split']]

# limit_nums = None

vqa_reader_train_cfg = dict(
    tasks=task_ids,
    bert_model='bert-base-uncased',
    do_lower_case=True,
    in_memory=False,  # whether use chunck for parallel training
    clean_datasets=True,  # whether clean train sets for multitask data
    is_train=True,
    # limit_nums=limit_nums,
    TASKS=TASKS,
)

vqa_reader_test_cfg = dict(
    tasks=task_ids,
    bert_model='bert-base-uncased',
    do_lower_case=True,
    in_memory=False,
    clean_datasets=True,
    is_train=False,
    # limit_nums=limit_nums,
    TASKS=TASKS,
)

train_data = dict(
    samples_per_gpu=TASKS['TASK' + task_ids]['per_gpu_train_batch_size'],
    workers_per_gpu=0,
    data=dict(
        type=dataset_type,
        reader=vqa_reader_train_cfg,
    ),
    pin_memory=True,
    sampler=TASKS['TASK' + task_ids]['sampler'],
)

test_data = dict(
    samples_per_gpu=TASKS['TASK' + task_ids]['per_gpu_eval_batch_size'],
    workers_per_gpu=4,
    data=dict(type=dataset_type, reader=vqa_reader_test_cfg),
    pin_memory=True,
)

post_processor = dict(
    type='Evaluator',
    metrics=[dict(type='VILBERT_AccuracyMetric')],
    dataset_converters=[dict(type='VILBERT_DatasetConverter')])
