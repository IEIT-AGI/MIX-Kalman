# model settings
model = dict(
    type='BUTD',
    embedding=dict(
        type='WordEmbedding',
        vocab_file='/home/datasets/mix_data/mixk/data/datasets/textvqa/defaults/extras/vocabs/vocabulary_100k.txt',
        embedding_dim=300),
    encoder=dict(
        type='ImageFeatureEncoder',
        encoder_type='finetune_faster_rcnn_fpn_fc7',
        in_dim=2048,
        weights_file='/home/datasets/mix_data/mixk/data/models/detectron.vmb_weights/fc7_w.pkl',
        bias_file='/home/datasets/mix_data/mixk/data/models/detectron.vmb_weights/fc7_b.pkl'),
    backbone=dict(
        type='ImageFeatureEmbedding',
        img_dim=2048,
        question_dim=2048,
        modal_combine=dict(
            type='top_down_attention_lstm', params=dict(dropout=0.5, hidden_dim=1024, attention_dim=1024)),
        normalization='softmax',
        transform=dict(type='linear', params=dict(out_dim=1))),
    head=dict(
        type='LanguageDecoder',
        in_dim=2048,
        out_dim=8001,  # TODO vocab_size
        dropout=0.5,
        hidden_dim=1024,
        feature_dim=2048,
        fc_bias_init=0,
        loss_cls=dict(type='CaptionCrossEntropyLoss')))
