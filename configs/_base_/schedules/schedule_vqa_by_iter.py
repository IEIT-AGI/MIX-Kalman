# optimizer  transform.AdamW
optimizer = dict(type='AdamW', lr=0.00005, weight_decay=0, eps=1e-9, betas=[0.9, 0.98], training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    use_warmup=True,
    lr_steps=[90000, 108000],
    lr_ratio=0.2,
    warmup_factor=0.25,
    warmup_iterations=27000,
    policy='MultiStepScheduler')
max_iter = 118000
max_iter = 236000
by_iter = True
