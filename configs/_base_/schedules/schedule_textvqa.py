optimizer = dict(type='Adam', lr=1e-4, weight_decay=0, eps=1e-8, training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=dict(
    clip_norm_mode='all',
    max_grad_l2_norm=0.25,
    use_scale=True,
))

lr_config = dict(
    use_warmup=True,
    lr_steps=[14000, 19000],
    lr_ratio=0.1,
    warmup_factor=0.2,
    warmup_iterations=1000,
    policy='PythiaScheduler')

total_epochs = 22
