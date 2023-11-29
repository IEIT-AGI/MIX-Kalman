# To finetuning the base model with dense annotations
_base_ = [
    '../_base_/models/visual-dialog-bert/basemodel+dense.py',
    '../_base_/datasets/visual-dialog/visual_dialog_with_dense_annotations.py',
    '../_base_/schedules/visual-dialog-bert/basemodel_dense_schedule.py',
    '../_base_/default_runtime/basemodel+dense.py'
]  # yapf:disable
