eval_iter_period = 1000
checkpoint_config = dict(iter_period=eval_iter_period)
log_config = dict(period=5)  # PeriodicLogger parameter
work_dir = './work_dirs'  # the dir to save logs and models

CUDNN_BENCHMARK = False
model_device = 'cuda'
find_unused_parameters = True
