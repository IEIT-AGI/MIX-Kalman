# optimizer  transform.AdamW
optimizer = dict(type='Adam', lr=0.0002, weight_decay=0.0001, training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    use_warmup=True,
    lr_steps=[400000, 808000],
    lr_ratio=0.2,
    warmup_factor=0.25,
    warmup_iterations=27000,
    policy='MultiStepScheduler')

total_epochs = 30
