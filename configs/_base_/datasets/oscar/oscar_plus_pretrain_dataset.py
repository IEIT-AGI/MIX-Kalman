dataset_type = 'OscarPretrainDataset'
data_root = '/home/datasets/mix_data/vinvl/pretrain/'

train_datasets = ['train']  # 'train+val'
test_datasets = ['val']  # 'test2015', 'test-dev2015',

params = {
    'data_dir': '/home/datasets/mix_data/vinvl/pretrain/',
    'dataset_file': '/home/datasets/mix_data/vinvl/pretrain/coco_flickr30k_googlecc_gqa_sbu_oi_x152c4big2exp168.yaml',
    'extra_dataset_file': None,
    'bert_model': 'bert',
    'output_dir': './testtrain',
    'chunk_start_id': -1,
    'chunk_end_id': -1,
    'max_img_seq_length': 50,
    'img_feature_dim': 2054,
    'img_feature_type': 'faster_r-cnn',
    'use_layernorm': False,
    'drop_out': 0.1,
    'use_b': 1,
    'textb_sample_mode': 1,  # 0
    'extra_textb_sample_mode': 1,
    'texta_false_prob': 0.25,  # 0
    'model_name_or_path': 'bert-base-uncased',
    'config_name': '',
    'tokenizer_name': '',
    'cache_dir': '',
    'max_seq_length': 35,
    'do_train': True,
    'learning_rate': 5e-05,
    'max_iters': 2000000,
    'train_batch_size': 8,
    'num_workers': 6,
    'adam_epsilon': 1e-08,
    'optim': 'adamw',
    'max_grad_norm': 10.0,
    'warmup_steps': 0,
    'no_cuda': False,
    'on_memory': True,
    'do_lower_case': True,
    'local_rank': -1,
    'seed': 42,
    'gradient_accumulation_steps': 1,
    'from_scratch': False,
    'use_img_layernorm': 1,
    'img_layer_norm_eps': 1e-12,
    'gpu_ids': '-1',
    'mask_loss_for_unmatched': 1,
    'extra_loss_weight': 0.0,
    'use_gtlabels': 1,
    'ckpt_period': 10000,
    'log_period': 100,
    'num_gpus': 1,
    'distributed': False,
    'n_gpu': 1,
    'num_contrast_classes': 2,
    'finetuning_task': None,
    'num_labels': 2,
    'output_attentions': False,
    'output_hidden_states': False,
    'torchscript': False,
    'pruned_heads': {},
    'vocab_size': 30522,
    'hidden_size': 768,
    'num_hidden_layers': 12,
    'num_attention_heads': 12,
    'hidden_act': 'gelu',
    'intermediate_size': 3072,
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'max_position_embeddings': 512,
    'type_vocab_size': 2,
    'initializer_range': 0.02,
    'layer_norm_eps': 1e-12,
    'architectures': ['BertForMaskedLM'],
    'model_type': 'bert',
    'pad_token_id': 0
}

train_cfg = dict(
    token=dict(name='bert-base-uncased', do_lower_case=True),
    oscar_tsv_dataset=dict(
        dataset_yaml_file=data_root + 'coco_flickr30k_googlecc_gqa_sbu_oi_x152c4big2exp168.yaml',
        on_memory=True,
        args=params,
    ),
    extra_dataset=None)

train_data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    data=dict(
        type=dataset_type,
        reader=train_cfg,
    ),
    collate_fn='lambda x: list(zip(*x))'
    # sampler='DistributedSampler',
)
