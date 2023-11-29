_base_ = [
    '../_base_/models/vinvl/vinvl_pretrain_config.py',
    '../_base_/datasets/oscar/oscar_plus_pretrain_dataset.py',  # oscar+ -> vinvl
    '../_base_/default_runtime.py',
]
