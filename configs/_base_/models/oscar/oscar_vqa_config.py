# model settings
model = dict(
    type='OSCAR',
    params=dict(
        img_feature_dim=2054,
        img_feature_type='faster_r-cnn',
        model_type='bert',
        model_name_or_path='/home/datasets/mix_data/model/oscar/base-vg-labels/ep_107_1192087',
        task_name='vqa_text',
        drop_out=0.3,
        loss_type='bce',
        img_feat_format='pt',
        classifier='linear',
        cls_hidden_scale=3,
        code_voc=512,
        code_level='top',  # code level: top, botttom, both
        num_labels=3129,
        config_name=None,
        training_head_type='vqa2',
        bert_model_name='bert-base-uncased',
        # fp16=False,
        # fp16_opt_level='O1',
        # local_rank=-1,
        # no_cuda=False,
        # adjust_dp=False,
        # adjust_loss=False,
        # adjust_loss_epoch=-1,
    ))

loss = dict(
    type='OSCARLoss', cfg=dict(
        loss_type='bce',
        num_labels=3129,
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
    num_training_steps=123950,  # ceil(totoal 634516 / batch size 32 / GPUS 4) * epoch size 25
    policy='WarmupLinearSchedule',
)

# by_iter = True
total_epochs = 25

eval_iter_period = 4000
checkpoint_config = dict(iter_period=eval_iter_period)

seed = 88
