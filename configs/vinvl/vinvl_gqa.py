_base_ = [
    '../_base_/models/oscar/oscar_gqa_config.py',
    '../_base_/datasets/oscar/oscar_gqa_dataset.py',
    '../_base_/default_runtime.py',
]

#  cover the parrmeter in above files
model = dict(params=dict(model_name_or_path='/home/datasets/mix_data/model/vinvl/vqa/base/checkpoint-2000000', ))

data_root = '/home/datasets/mix_data/vinvl/datasets/gqa'

gqa_reader_train_cfg = dict(
    model_name_or_path='/home/datasets/mix_data/model/vinvl/vqa/base/checkpoint-2000000',
    data_dir=data_root,
    label_file=data_root + '/trainval_testdev_all_ans2label.pkl',
    label2ans_file=data_root + '/trainval_testdev_all_label2ans.pk',
)

gqa_reader_test_cfg = dict(
    model_name_or_path='/home/datasets/mix_data/model/vinvl/vqa/base/checkpoint-2000000',
    data_dir=data_root,
    label_file=data_root + '/trainval_testdev_all_ans2label.pkl',
    label2ans_file=data_root + '/trainval_testdev_all_label2ans.pk',
)

train_data = dict(data=dict(reader=gqa_reader_train_cfg, ), )

test_data = dict(data=dict(reader=gqa_reader_test_cfg, ), )
