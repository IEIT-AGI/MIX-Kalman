# model settings for VCR
model = dict(
    type='UNITER_VCR',
    params=dict(
        # num_labels=3129,
        model_config='configs/_base_/models/uniter/uniter-base.json',
        dropout=0.1,
        num_special_tokens=81,
        img_dim=2048,
        checkpoint_from='vcr_pretrain',
        pretrained_path='/home/datasets/mix_data/UNITER/vcr/uniter-base-vcr_2nd_stage.pt',
    ),
)

loss = dict(type='CrossEntropyLoss', params=dict(reduction='mean'))

optimizer = dict(
    type='TansformerAdamW',
    constructor='UniterVQAOptimizerConstructor',
    paramwise_cfg=dict(
        weight_decay=0.01,
        lr_mul=1.0,
        key_named_param='vcr_output',
    ),
    lr=6e-5,
    betas=[0.9, 0.98],
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=2.0))

# fp16 = dict(
#     init_scale=2.**16,
#     growth_factor=2.0,
#     backoff_factor=0.5,
#     growth_interval=2000,
# )

lr_config = dict(
    num_warmup_steps=800,
    num_training_steps=8000,
    policy='WarmupLinearSchedule',
)

total_epochs = 4

eval_iter_period = 1000
checkpoint_config = dict(iter_period=eval_iter_period)

gradient_accumulation_steps = 5
is_lr_accumulation = True

seed = 42
