from configs._base_.datasets.vilbert.vilbert_task_config import (
    task_ids,
    TASKS,
)

# model settings
model = dict(
    type='VILBERT',
    params=dict(
        # below from bert_base_6layer_6conect.json
        bi_hidden_size=1024,
        bi_num_attention_heads=8,
        bi_intermediate_size=1024,
        bi_attention_type=1,
        pooling_method='mul',
        visual_target=0,  # which target to use for visual branch 0: soft label, 1: regress the feature, 2: NCE loss."
        fast_mode=False,
        fixed_v_layer=0,
        fixed_t_layer=0,
        in_batch_pairs=False,
        fusion_method='mul',
        dynamic_attention=False,
        with_coattention=True,
        objective=0,
        num_negative=128,
        model='bert',
        task_specific_tokens=True,
        visualization=False,
        t_config=dict(
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=12,
            type_vocab_size=2,
            vocab_size=30522,
            biattention_id=[6, 7, 8, 9, 10, 11],
            layer_norm_eps=1e-12,
            task_specific_tokens=True,
        ),
        v_config=dict(
            feature_size=2048,
            target_size=1601,
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            initializer_range=0.02,
            biattention_id=[0, 1, 2, 3, 4, 5],
        ),
        # below from parse argument
        tasks=task_ids,  # '1-2-3...' training task separate by -
        bert_model='bert-base-uncased',  # 'roberta'
        from_pretrained='/home/datasets/mix_data/model/vilbert/multi_task_model.bin',
        train_iter_multiplier=1,  # multiplier for the multi-task training
        # forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop
        train_iter_gap=4,
        do_lower_case=True,
        gradient_accumulation_steps=1,
        freeze=-1,  # till which layer of textual stream of vilbert need to fixed
        vision_scratch=False,  # whether pre-trained the image or not.
        fp16=False,  # Whether to use 16-bit float precision instead of 32-bit
        TASKS=TASKS,
        training_head_type='vqa2',
    ))

loss = dict(
    type='VILBERTMutilLoss', task_cfg=dict(
        tasks=task_ids,
        TASKS=TASKS,
    ))

optimizer = dict(
    type='TansformerAdamW',
    constructor='VilbertOptimizerConstructor',
    paramwise_cfg=dict(
        language_weights_file='/home/datasets/mix_data/model/vilbert/config/bert-base-uncased_weight_name.json',
        vision_scratch=False,  # whether pre-trained the image or not.
    ),
    lr=TASKS['TASK' + task_ids]['lr'],
    correct_bias=False,
    training_encoder_lr_multiply=1)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    num_warmup_steps=TASKS['TASK' + task_ids]['num_warmup_steps'],
    num_training_steps=TASKS['TASK' + task_ids]['num_training_steps'],
    policy='WarmupLinearSchedule')

# by_iter = True
total_epochs = 20
'''
fp16 = dict(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)
'''

seed = 0
