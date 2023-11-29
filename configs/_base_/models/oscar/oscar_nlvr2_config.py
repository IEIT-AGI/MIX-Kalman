# model settings
model = dict(
    type='OSCAR_NLVR',
    params=dict(
        num_labels=2,
        classifier='mlp',
        cls_hidden_scale=3,
        code_level='top',
        code_voc=512,
        config_name=None,
        drop_out=0.3,
        model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
        model_type='bert',
        num_choice=2,
        task_name='nlvr',
        tokenizer_name=None,
        use_layernorm=False,
        use_pair=True,
        img_feature_dim=2054,
        img_feature_type='faster_r-cnn',
        loss_type='xe',
        training_head_type='vqa2',
        bert_model_name='bert-base-uncased',
        # fp16=False,
        # fp16_opt_level='O1',
        # local_rank=-1,
        # no_cuda=False,
    ))

loss = dict(
    type='OSCARLoss', cfg=dict(
        loss_type='xe',
        num_labels=2,
    ))

optimizer = dict(
    type='TansformerAdamW',
    constructor='OscarOptimizerConstructor',
    paramwise_cfg=dict(weight_decay=0.05, ),
    lr=3e-05,
    eps=1e-8,
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

lr_config = dict(
    num_warmup_steps=10000,  # warmup_proportion=0
    num_training_steps=24000,  # ceil(totoal 86373 / batch size 72 / GPUS 1) * epoch size 20
    policy='WarmupLinearSchedule',
)

# by_iter = True
total_epochs = 20

eval_iter_period = 4000
checkpoint_config = dict(iter_period=eval_iter_period)

seed = 88
