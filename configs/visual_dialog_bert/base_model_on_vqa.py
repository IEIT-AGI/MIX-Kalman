# To train the base model(no finetuning on dense annotations)
_base_ = [
    '../_base_/models/visual-dialog-bert/basemodel.py',
    '../_base_/datasets/visual-dialog/visual_dialog_dataset.py',
    '../_base_/schedules/visual-dialog-bert/basemodel_schedule.py',
    '../_base_/default_runtime/basemodel.py'
]  # yapf:disable
