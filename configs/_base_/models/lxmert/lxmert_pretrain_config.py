model = dict(
    type='LXMERT_Pretrain',
    params=dict(
        bert_model_name='bert-base-uncased',
        num_answers=9500,
        max_seq_length=20,
        task_mask_lm=True,
        task_obj_predict=True,
        task_matched=True,
        task_qa=True,
        visual_losses='obj,attr,feat',
        freeze_base=False,
        llayers=9,
        xlayers=5,
        rlayers=5,
        from_scratch=True,
        load=None,
        load_lxmert=None,
    ))

loss = dict(type='PretrainLoss')

optimizer = dict(
    type='BertAdam',
    lr=1e-4,
    warmup=0.05,
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=1))
lr_config = dict(
    warmup=0.1,
    warmup_method='warmup_linear',
    # max_iters=117876,  # ceil(totoal 942999 / batch size 32) * epoch size datasets: train
    max_iters=134380,  # floor(totoal 1075062 / batch size 32) * epoch size datasets: train, valid
    policy='BertWarmupLinearLR')

total_epochs = 4
seed = 9595
