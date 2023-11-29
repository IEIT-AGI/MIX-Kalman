# model settings
model = dict(
    type='OSCAR_GQA',
    params=dict(
        num_labels=1853,
        classifier='linear',
        cls_hidden_scale=2,
        code_voc=512,
        config_name=None,
        drop_out=0.3,
        img_feature_dim=2054,
        img_feature_type='faster_r-cnn',
        loss_type='xe',
        model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
        model_type='bert',
        spatial_dim=6,
        task_name='gqa',
        tokenizer_name=None,
        training_head_type='vqa2',
        bert_model_name='bert-base-uncased',
        # code_level='top',
        # fp16=False,
        # fp16_opt_level='O1',
        # local_rank=-1,
        # no_cuda=False,
    ))

loss = dict(
    type='OSCARLoss', cfg=dict(
        loss_type='xe',
        num_labels=1853,
    ))

optimizer = dict(
    type='TansformerAdamW',
    constructor='OscarOptimizerConstructor',
    paramwise_cfg=dict(weight_decay=0.05),
    lr=5e-05,
    eps=1e-8,
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

lr_config = dict(
    num_warmup_steps=0,  # warmup_proportion=0
    num_training_steps=424930,  # ceil(totoal 16317209 / batch size 48 / GPUS 4) * epoch size 5
    policy='WarmupLinearSchedule',
)

# by_iter = True
total_epochs = 5

eval_iter_period = 4000
checkpoint_config = dict(iter_period=eval_iter_period)

seed = 88
