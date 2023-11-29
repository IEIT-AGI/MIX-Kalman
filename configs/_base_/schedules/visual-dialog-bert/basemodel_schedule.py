optimizer = dict(
    type='AdamW',
    constructor='BertOptimizerConstructor',
    paramwise_cfg=dict(
        language_weights_file='~/mixk/mixk/mixk/models/visual_dialog_model/config/language_weights.json'),
    lr=1e-5,  # 2e-5
    image_lr=1e-5,  # learning rate for vision params
    training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='WarmupLinearScheduleNonZero',
    use_warmup=True,
    warmup_iterations=10000,  # 10000
    t_total=200000,  # 200000
)

total_epochs = 10
