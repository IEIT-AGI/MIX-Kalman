data_root = '/home/datasets/mix_data/mixk/'
vocab_path = 'data/datasets/textvqa/defaults/extras/vocabs/'
vocab_file = data_root + vocab_path + 'fixed_answer_vocab_textvqa_5k.txt'
weight_root = '/home/datasets/mix_data/mixk/data/models/detectron.vmb_weights/'

model = dict(
    type='M4C',
    m4c_config=dict(
        lr_scale_frcn=0.1,
        lr_scale_text_bert=0.1,
        lr_scale_mmt=1.0,  # no scaling
        text_bert_init_from_bert_base=True,
        text_bert=dict(num_hidden_layers=3),
        obj=dict(
            type='ImageFeatureEncoder',
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file=weight_root + 'fc7_w.pkl',
            bias_file=weight_root + 'fc7_b.pkl',
            dropout_prob=0.1,
            mmt_in_dim=2048,
        ),
        ocr=dict(
            type='ImageFeatureEncoder',
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file=weight_root + 'fc7_w.pkl',
            bias_file=weight_root + 'fc7_b.pkl',
            mmt_in_dim=3002,  # 300 (FastText) + 604 (PHOC) + 2048 (Faster R-CNN) + 50 (all zeros; legacy)
            dropout_prob=0.1),
        mmt=dict(hidden_size=768, num_hidden_layers=4),
        classifier=dict(
            type='linear', ocr_max_num=50, ocr_ptr_net=dict(
                hidden_size=768,
                query_key_size=768,
            ), params=dict()),
        # model_data_dir=weight_root,
        head=dict(type='LinearHead', in_dim=768, out_dim=5000),
        answer_processor=dict(type='TextVQAAnswerProcessor', vocab_file=vocab_file)),
)

loss = dict(type='M4CDecodingBCEWithMaskLoss')
