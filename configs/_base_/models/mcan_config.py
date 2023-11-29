model = dict(
    type='MCAN',
    embedding=[
        dict(
            type='WordEmbedding',
            vocab_file='~/.cache/torch/mixk/data/datasets/textvqa/defaults/extras/vocabs/vocabulary_100k.txt',
            embedding_dim=300,
            glove_params=dict(
                name='6B',
                dim=300,
                cache='/home/datasets/mix_data/mixk',
            )),
        dict(
            type='TextEmbedding',
            emb_type='mcan',
            hidden_dim=1024,
            embedding_dim=300,
            num_attn=8,
            dropout=0.1,
            num_layers=6,
            num_attn_pool=1,
            num_feat=2)
    ],
    encoder=dict(type='ImageFeatureEncoder', encoder_type='default'),
    backbone=dict(
        type='TwoBranchEmbedding',
        embedding_dim=2048,
        hidden_dim=1024,
        cond_dim=1024,
        num_attn=8,
        dropout=0.1,
        num_layers=6,
        cbn_num_layers=4),
    combine_model=dict(type='BranchCombineLayer', img_dim=1024, ques_dim=1024),
    head=dict(
        type='TripleLinearHead',
        in_dim=2048,
        out_dim=3129,
    ))
loss = dict(type='TripleLogitBinaryCrossEntropy')
