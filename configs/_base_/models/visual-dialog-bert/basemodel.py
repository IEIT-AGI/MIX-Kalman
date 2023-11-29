model = dict(
    type='VisDiaBERT',
    config=dict(
        pretrained_model_name_or_path='/home/datasets/mix_data/torch/pytorch_transformers/bert/bert-base-uncased',
        bert_file_path='~/mixk/mixk/mixk/models/visual_dialog_model/config/bert_base_6layer_6conect.json',
        sample_size=8,  # equal to batch_size*2   40
    ))
loss = dict(
    type='VisualDialogBertLoss',
    MLM_loss=dict(type='CrossEntropyLoss', weight_coeff=1, params=dict(ignore_index=-1)),
    # masked language modeling loss
    NSP_loss=dict(type='CrossEntropyLoss', weight_coeff=1, params=dict(ignore_index=-1)),
    # next sentence prediction loss
    MIR_loss=dict(type='KLDivLoss', weight_coeff=1, params=dict(reduction='none')),  # masked image region loss
)
