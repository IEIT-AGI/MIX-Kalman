# model settings
model = dict(
    type='LCGN',
    encoder=dict(
        type='LCGNEncoder',
        WRD_EMB_INIT_FILE='~/.cache/torch/checkpoints/gloves_gqa_no_pad.npy',
        encInputDropout=0.8,
        qDropout=0.92,
        WRD_EMB_DIM=300,
        ENC_DIM=512,
        WRD_EMB_FIXED=False),
    backbone=dict(
        type='LCGN_BACKBONE',
        stem_linear=True,
        D_FEAT=2112,
        CTX_DIM=512,
        CMD_DIM=512,
        MSG_ITER_NUM=4,
        stemDropout=1.0,
        readDropout=0.85,
        memoryDropout=0.85,
        CMD_INPUT_ACT='ELU',
        STEM_NORMALIZE=True),
    head=dict(
        type='LCGNClassiferHead',
        in_dim=512,
        out_dim=1845,
        OUT_QUESTION_MUL=True,
        CMD_DIM=512,
        outputDropout=0.85,
    ))

loss = dict(type='LogitBinaryCrossEntropy')
