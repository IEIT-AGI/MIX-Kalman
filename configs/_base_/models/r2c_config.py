# model settings
model = dict(
    type='R2C',
    input_dropout=0.3,
    pretrained=True,
    average_pool=True,
    semantic=True,
    final_dim=512,
    backbone=dict(
        type='R2C_BACKBONE',
        input_dropout=0.3,
        reasoning_use_obj=True,
        reasoning_use_answer=True,
        reasoning_use_question=True,
        pool_reasoning=True,
        pool_answer=True,
        pool_question=True),
    head=dict(
        type='R2CHead',
        in_dim=1536,
        out_dim=1024,
        dropout=0.3,
    ))

loss = [dict(type='CrossEntropyLoss'), dict(type='OBJCrossEntropyLoss')]
