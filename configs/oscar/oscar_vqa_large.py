_base_ = [
    '../_base_/models/oscar/oscar_vqa_config.py',
    '../_base_/datasets/oscar/oscar_vqa_dataset.py',
    '../_base_/default_runtime.py',
]

#  cover the parrmeter in above files
model = dict(params=dict(model_name_or_path='/home/datasets/mix_data/model/oscar/large-vg-labels/ep_20_590000', ))

optimizer = dict(lr=3e-05, )

lr_config = dict(
    num_training_steps=167950,  # ceil(totoal 644918 / batch size 24 / GPUS 4) * epoch size 25
)

train_datasets = ['train+val']  # 'train'

vqa_reader_train_cfg = dict(
    model_name_or_path='/home/datasets/mix_data/model/oscar/large-vg-labels/ep_20_590000',
    name=train_datasets,
)

vqa_reader_test_cfg = dict(model_name_or_path='/home/datasets/mix_data/model/oscar/large-vg-labels/ep_20_590000', )

train_data = dict(
    samples_per_gpu=24,
    data=dict(reader=vqa_reader_train_cfg, ),
)

test_data = dict(data=dict(reader=vqa_reader_test_cfg, ), )
