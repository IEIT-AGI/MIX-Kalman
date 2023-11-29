_base_ = [
    '../_base_/models/oscar/oscar_vqa_config.py',
    '../_base_/datasets/oscar/oscar_vqa_dataset.py',
    '../_base_/default_runtime.py',
]

#  cover the parrmeter in above files
model = dict(params=dict(model_name_or_path='/home/datasets/mix_data/model/vinvl/vqa/base/checkpoint-2000000', ))

data_root = '/home/datasets/mix_data/vinvl/datasets/vqa'

vqa_reader_train_cfg = dict(
    model_name_or_path='/home/datasets/mix_data/model/vinvl/vqa/base/checkpoint-2000000',
    data_dir=data_root,
    label_file=data_root + '/trainval_ans2label.pkl',
    txt_data_dir=data_root,
)

vqa_reader_test_cfg = dict(
    model_name_or_path='/home/datasets/mix_data/model/vinvl/vqa/base/checkpoint-2000000',
    data_dir=data_root,
    label_file=data_root + '/trainval_ans2label.pkl',
    txt_data_dir=data_root,
)

train_data = dict(data=dict(reader=vqa_reader_train_cfg, ), )

test_data = dict(data=dict(reader=vqa_reader_test_cfg, ), )
