optimizer = dict(type='AdamW', lr=0.00005, weight_decay=0, eps=1e-9, betas=[0.9, 0.98], training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)
fp16 = dict(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)

lr_config = dict(
    use_warmup=True,
    lr_steps=[90000, 108000],
    lr_ratio=0.2,
    warmup_factor=0.25,
    warmup_iterations=27000,
    policy='MultiStepScheduler')

total_epochs = 8
