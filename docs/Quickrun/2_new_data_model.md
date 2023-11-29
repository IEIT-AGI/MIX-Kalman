# 2: Train with customized datasets

The basic steps are as below:

1. Prepare the customized dataset;
2. Prepare other necessary configs;
3. Train, test, inference models on the customized dataset.

In this part, we use [LXMERT](https://github.com/IEIT-AGI/MIX-Kalman/tree/master/configs/lxmert/LXMERT.md) model and VQA task as an example.

## Prepare the customized dataset

We provide you with two choices when you customize dataset: use MIX-Kalman or directly use source code data process module. In this section, we only introduce how to customized dataset in MIX-Kalman. For more details, please refer to the [Tutorial2](../tutorials/Tutorial2-customize_dataset.md).

### Download dataset files

In MIX-Kalman, we use the extracted feature files refer to the source code papers(i.e. VQA dataset).

Download the feature and annotation files like `train.json`, `trainval_ans2label.json`, `train2014_obj36.tsv` and so on. You can download the data file according to the original paper or source code of [LXMERT](https://github.com/IEIT-AGI/MIX-Kalman/tree/master/configs/lxmert/LXMERT.md).

Put all those files outside the project directory and symlink the dataset root to `MIX-Kalman/configs/_base_/datasets/lxmert/lxmert_vqa.py`

### Prepare the dataset config file

You should set the train data, test data path and so on in the config file, thus it can be loaded successfully.

```python
# set dataset type for registry
dataset_type = 'VQATorchDataset'
# set dataset path for root, feature, annotation, vocab, etc
data_root = '/home/datasets/mix_data/lxmert/'
feature_path = 'mscoco_imgfeat/'
annotation_path = 'vqa/'
...
# set default parameter and train data for vqa_reader.py
vqa_reader_train_cfg = dict(
    annotations=dict(
        train=data_root + annotation_path + 'train.json',
        ...),
    ...
)

# set default config for train data
train_data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    data=dict(
        type=dataset_type,
        reader=vqa_reader_train_cfg,
        # limit_nums=400,
    ),
    drop_last=True,
    shuffle=True,)
```

### Add dataset process module

1. add dataset_loader.py
   - the class `VQADATASET` should inherit from the parent class `BaseLoader` and register the dataset by decorator `@DATASETS.register_module()`;
   - writer the specific item dictionary in the `__getitem__()` function according to different task.
2. add dataset_reader.py
   - the `VQAReader` should inherit from the parent class `mixkDataReader` . It will read the parameters of the `vqa_reader_train_cfg` in the parent class;
   - writer the `__getitem__()` function according to different task.
3. add dataset_infocpler.py (some datasets don't need infocpler.py)
   - the `VQAInfoCpler` should inherit from the parent class `BaseInfoCpler` . It will read the parameters of the `vqa_info_cpler_cfg` in the parent class;
   - writer `completeInfo` function according to different task.

## Prepare other necessary configs

There are three configs you should prepare:

1. Prepare model config file

   You should modify the output dimension in the model config file to match with the class numbers of the new datasets and set the pre-trained model weight path in the `from_pretrained` or `--load-from`.

   ```python
   model = dict(
       type='LXMERT',
       params=dict(
           num_labels=3129,  # set the output dimension same with the dataset class
          pretrained_path='/home/datasets/mix_data/MIX-Kalman/data/models/model_LXRT.pth',
       ...
       ))
   ```

   For more detailed usages, please refer to the [Tutorial6](../tutorials/Tutorial6-finetune.md).

2. prepare the schedule and runtime config

   You could modify some default settings like the directory path to save logs and models and so on.

   For more detailed usages, please refer to the [Tutorial4](../tutorials/Tutorial4-customize_Schedule_and_Runtime_Settings.md).

3. Prepare base configs

   You should set the configuration path of the model, dataset, and default runtime in the `configs` directory. (e.g. [LXMERT model and VQA task](https://github.com/IEIT-AGI/MIX-Kalman/tree/master/configs/lxmert/lxmert_vqa.py)).

   ```python
   _base_ = [
       '../_base_/models/lxmert/lxmert_vqa_config.py',
       '../_base_/datasets/lxmert/lxmert_vqa.py',
       '../_base_/default_runtime.py',
   ]
   ```


## Train model

To train a model with the new config, you can simply run

```shell
python tools/run.py --config-file configs/lxmert/lxmert_vqa.py --resume-from ./work_dirs/lxmert_vqa.pth
```

For more detailed usages, please refer to the [Case 1](1_exist_data_model.md).

## Test and inference

To test the trained model, you can simply run

```shell
python tools/run.py --config-file configs/lxmert/lxmert_vqa.py --load-from ./work_dirs/lxmert_vqa.pth --eval-only
```

For more detailed usages, please refer to the [Case 1](1_exist_data_model.md).
