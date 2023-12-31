# eval_iter_period = 500
# checkpoint_config = dict(iter_period=eval_iter_period)
log_config = dict(period=1)  # PeriodicLogger parameter
work_dir = './work_dirs'  # the dir to save logs and models

# load_from = '/home/datasets/mix_data/model/visdial_model_mixk/vqa_weights.pth'
# load_from = '~/mixk/mixk/work_dirs/epoch18_model.pth'

# resume_from = '~/mixk/mixk/tools/work_dirs/11/OSCAR_OSCAR_VQADataset_epoch14_model.pth'
seed = 13
CUDNN_BENCHMARK = False
model_device = 'cuda'
find_unused_parameters = True

# gradient_accumulation_steps = 5
# is_lr_accumulation = True
