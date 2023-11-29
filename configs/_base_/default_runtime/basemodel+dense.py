log_config = dict(period=5)
work_dir = './work_dirs'  # the dir to save logs and models
CUDNN_BENCHMARK = False
model_device = 'cuda'
find_unused_parameters = True
gradient_accumulation_steps = 10
is_lr_accumulation = False
