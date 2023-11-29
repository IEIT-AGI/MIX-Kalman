from configs._base_.datasets.devlbert.devlbert_task_config import (
    task_ids,
    TASKS,
)

model = dict(
    type='DEVLBERT',
    params=dict(
        # below from bert_base_6layer_6conect.json
        bi_hidden_size=1024,
        bi_num_attention_heads=8,
        bi_intermediate_size=1024,
        bi_attention_type=1,
        pooling_method='mul',
        predict_feature=False,
        fast_mode=False,
        fixed_v_layer=0,
        fixed_t_layer=0,
        in_batch_pairs=False,
        fusion_method='mul',
        intra_gate=False,
        with_coattention=True,
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
        bert_model='bert-base-uncased',
        from_pretrained='/home/datasets/mix_data/DeVLBert/pytorch_model_11.bin',
        train_iter_multiplier=1,  # multiplier for the multi-task training
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
    type='BertAdam',
    constructor='DevlbertOptimizerConstructor',
    paramwise_cfg=dict(
        language_weights_file='/home/datasets/mix_data/DeVLBert/bert_file/bert-base-uncased_weight_name.json',
        vision_scratch=False,  # whether pre-trained the image or not.
    ),
    lr=TASKS['TASK' + task_ids]['lr'],
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=None)

iters_in_epoch = max([TASKS['TASK' + task_id]['iters_in_epoch'] for task_id in task_ids.split('-')])
lr_reduce_list = [12, 16]  # epoch
warmup_constant_lr_task = ['TASK2', 'TASK3']
lr_config = dict(
    num_warmup_steps=TASKS['TASK' + task_ids]['num_warmup_steps'],
    policy='WarmupConstantSchedule',
) if any('TASK' + task_id in warmup_constant_lr_task for task_id in task_ids.split('-')) else dict(
    milestones=[iters_in_epoch * k for k in lr_reduce_list],
    gamma=0.1,
    warmup_factor=0,
    warmup_iters=TASKS['TASK' + task_ids]['num_warmup_steps'],
    policy='WarmupMultiStepLR',
)

# by_iter = True
total_epochs = TASKS['TASK' + task_ids]['num_epoch']

seed = 0
