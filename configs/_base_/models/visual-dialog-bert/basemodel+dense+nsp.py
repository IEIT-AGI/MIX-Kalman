model = dict(
    type='VisDiaBERT',
    config=dict(
        pretrained_model_name_or_path='/home/datasets/mix_data/torch/pytorch_transformers/bert/bert-base-uncased',
        bert_file_path='~/mixk/mixk/mixk/models/visual_dialog_model/config/bert_base_6layer_6conect.json',
        sample_size=10,  # equal to batch_size*2
        is_dense=True,  # dense annotations -> sample_size = 80
    ))

loss = dict(
    type='VisualDialogBertDenseLoss',
    NSP_loss=dict(type='CrossEntropyLoss', weight_coeff=1),
    KLDiv_loss=dict(type='KLDivLoss', weight_coeff=1, params=dict(reduction='batchmean')),
    MLM_loss=dict(type='CrossEntropyLoss', weight_coeff=0, params=dict(ignore_index=-1)),
    MIR_loss=dict(type='KLDivLoss', weight_coeff=0, params=dict(reduction='none')),  # masked image region loss
)
