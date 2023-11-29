_base_ = [
    '../_base_/models/oscar/oscar_nlvr2_config.py',
    '../_base_/datasets/oscar/oscar_nlvr2_dataset.py',
    '../_base_/default_runtime.py',
]

#  cover the parrmeter in above files
model = dict(
    params=dict(
        model_name_or_path='/home/datasets/mix_data/model/oscar/large-vg-labels/ep_55_1617000',
        cls_hidden_scale=2,
    ))

lr_config = dict(
    num_warmup_steps=5000,  # warmup_proportion=0
    num_training_steps=36000,  # ceil(totoal 86373 / batch size 24 / GPUS 2) * epoch size 20
)

nlvr_reader_train_cfg = dict(model_name_or_path='/home/datasets/mix_data/model/oscar/large-vg-labels/ep_55_1617000', )

nlvr_reader_test_cfg = dict(model_name_or_path='/home/datasets/mix_data/model/oscar/large-vg-labels/ep_55_1617000', )

train_data = dict(
    samples_per_gpu=24,
    data=dict(reader=nlvr_reader_train_cfg, ),
    sampler='DistributedSampler',
)

test_data = dict(data=dict(reader=nlvr_reader_test_cfg, ), )
