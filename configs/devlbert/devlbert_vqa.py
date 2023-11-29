_base_ = [
    '../_base_/models/devlbert/devlbert_config.py',
    '../_base_/datasets/devlbert/devlbert_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/custom_hook/devlbert_ema_hook.py',
]
