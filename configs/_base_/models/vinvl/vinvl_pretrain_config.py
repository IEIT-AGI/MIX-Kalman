# model settings
transformer_cache_dir = '/home/datasets/mix_data/torch/pytorch_transformers'
model = dict(
    type='OscarPreTraining',
    bert_cfg=dict(
        type='BertConfig',
        run_method=dict(
            type='from_pretrained', pretrained_model_name_or_path='bert-base-uncased', cache_dir=transformer_cache_dir),
        img_layer_norm_eps=1e-12,
        use_img_layernorm=1,
        img_feature_dim=2054,
        img_feature_type='faster_r-cnn',
        hidden_dropout_prob=0.1,
        texta_false_prob=0,
        use_b=1,
    ),
    pretrained_cfg=dict(
        type='BertImgForPreTraining',
        run_method=dict(
            type='from_pretrained',
            pretrained_model_name_or_path='bert-base-uncased',
            cache_dir=transformer_cache_dir,
        )),
)

loss = dict(type='PretrainLoss')

optimizer = dict(
    type='AdamW',
    constructor='OscarPreTrainOptimizerConstructor',
    paramwise_cfg=dict(weight_decay=0.01),
    lr=5e-05,
    eps=1e-8,
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

lr_config = dict(
    num_warmup_steps=0,  # warmup_proportion=0
    num_training_steps=123950,  # ceil(totoal 634516 / batch size 32 / GPUS 4) * epoch size 25
    policy='WarmupLinearSchedule',
)

total_epochs = 25
