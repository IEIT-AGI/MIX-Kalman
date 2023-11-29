# model settings
model = dict(
    type='LXMERT',
    params=dict(
        random_initialize=False,
        num_labels=3129,
        # BertConfig
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        #
        mode='lxr',
        l_layers=9,  # 12
        x_layers=5,  # 5
        r_layers=5,  # 0
        visual_feat_dim=2048,
        visual_pos_dim=4,
        freeze_base=False,
        max_seq_length=20,
        model='bert',
        training_head_type='vqa2',
        bert_model_name='bert-base-uncased',
        pretrained_path='/home/datasets/mix_data/mixk/data/models/model_LXRT.pth',
        label2ans_path='/home/datasets/mix_data/lxmert/vqa/trainval_label2ans.json',
    ))

loss = dict(type='LogitBinaryCrossEntropy')

optimizer = dict(
    type='BertAdam1',
    lr=5e-5,
    weight_decay=0.01,
    eps=1e-6,
    betas=[0.9, 0.999],
    max_grad_norm=-1,
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=5))
'''
fp16 = dict(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)
'''

lr_config = dict(
    warmup=0.1,
    warmup_method='warmup_linear',
    # max_iters=55472,  # ceil(totoal 443753 / batch size 32) * epoch size  datasets: train
    max_iters=79012,  # floor(totoal 632117 / batch size 32) * epoch size  datasets: train, nominival
    policy='BertWarmupLinearLR')
total_epochs = 4

seed = 9595
