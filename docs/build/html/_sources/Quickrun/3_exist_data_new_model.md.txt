# 3: Train with customized models and existing datasets

In this note, you will know how to train, test and inference your own customized models under existing datasets. We use the VQA dataset to train a customized MCAN model as an example to demonstrate the whole process.
The basic steps are as below:
1. Prepare the existing dataset;
2. Prepare your own customized model;
3. Prepare other necessary configs;
4. Train, test, and inference models on the standard dataset.

## Prepare dataset

The steps are as follows (i.e. VQA dataset):

- download the feature and annotation files like `train2014`, `imdb_train2014.npy`, `glove.6B.50d.txt.pt` and so on. You can download the data file according to the original paper or source code of [MCAN](https://github.com/IEIT-AGI/MIX-Kalman/tree/master/configs/mcan/MCAN.md);

- put all those files outside the project directory and symlink the dataset root to `MIX-Kalman/configs/_base_/datasets/vqa_dataset_grid_data.py` ;


See section [Prepare datasets](1_exist_data_model.md) for details.

## Prepare customized model

Assume that we want to implement a new neck called `BranchCombineLayer` under the existing model MCAN, which is as follows.

### Define a new combine_model (e.g. BranchCombineLayer)

Create a new file `mixk/models/combine_layers/branchcombinelayers.py`.

```python
from ..builder import COMBINE_LAYERS


@COMBINE_LAYERS.register_module()
class BranchCombineLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
```

### Import the module

You can add the following line to `mixk/models/combine_layers/__init__.py`, and add `'BranchCombineLayer'` in `__all__`.

```python
from .branchcombinelayers import BranchCombineLayer
__all__ = [
    'BranchCombineLayer', ...
]
```

### Modify the config file

```python
model = dict(
    ...
    combine_model=[
        dict(
        type='BranchCombineLayer',
        arg1=xxx,
        arg2=xxx,
        ...),
    ...
```

For more detailed usages about customize your own models (e.g. implement a new backbone, head, loss, etc) and runtime training settings (e.g. define a new optimizer, use gradient clip, customize training schedules and hooks, etc), please refer to the guideline [Customize Models](../tutorials/Tutorial3-customize_models.md) and [Customize Schedule and Runtime Settings](../tutorials/Tutorial4-customize_Schedule_and_Runtime_Settings.md) respectively.

## Prepare necessary configs

Then prepare necessary configs for your own training setting.

1. Prepare dataset config file

   You should set the train data, test data path and so on in the config file, thus it can be loaded successfully.

   ```python
   # set dataset type for registry
   dataset_type = 'VQADATASET'
   # set dataset path for root, feature, annotation, vocab, etc
   data_root = '/home/datasets/mix_data/MIX-Kalman/'
   feature_path = 'data/datasets/vqa2/grid_features/features/'
   annotation_path = 'data/datasets/vqa2/grid_features/annotations/'
   ...
   # set default parameter and train data for vqa_reader.py
   vqa_reader_train_cfg = dict(
       mix_features=dict(
           train=data_root + feature_default_path + 'trainval2014.lmdb',
           ...),
       ...
   )
   # set default parameter and test data for vqa_infocpler.py
   vqa_info_cpler_cfg = dict(...)

   # set default config for train data
   train_data = dict(...)
   ```

   For more detailed usages, please refer to the [Tutorial2](../tutorials/Tutorial2-customize_dataset.md).

2. Prepare model config file

   You should modify the `combine_model` type as your new component name and match parameters with  the input args.

   ```python
   model = dict(
       type='MCAN',
       embedding=[...],
       encoder=dict(...),
       backbone=dict(...),
       combine_model=dict(
           type='BranchCombineLayer',
           img_dim=1024,
           ques_dim=1024),
       head=dict(...))
   loss = dict(type='TripleLogitBinaryCrossEntropy')
   ```

   For more detailed usages, please refer to the [Tutorial3](../tutorials/Tutorial3-customize_models.md).

3. prepare the schedule and runtime config

   You could modify some default settings like the directory path to save logs and models and so on.

   For more detailed usages, please refer to the [Tutorial4](../tutorials/Tutorial4-customize_Schedule_and_Runtime_Settings.md).

4. Prepare base config

   You should set the configuration path of the model, dataset, schedule, and runtime in the `configs` directory. (e.g. [MCAN model and VQA task](https://github.com/IEIT-AGI/MIX-Kalman/tree/master/configs/mcan/mcan_vqa.py)).

   ```python
   _base_ = [
       '../_base_/models/mcan_config.py',
       '../_base_/datasets/vqa_dataset_grid_data.py',
       '../_base_/schedules/schedule_vqa.py',
       '../_base_/default_runtime.py'
   ]
   ```

## Train model

To train a model with the new config, you can simply run

```shell
python tools/run.py --config-file configs/mcan/mcan_vqa.py --resume-from ./work_dirs/mcan_vqa.pth
```

For more detailed usages, please refer to the [Case 1](1_exist_data_model.md).

## Test and inference

To test the trained model, you can simply run

```shell
python tools/run.py --config-file configs/mcan/mcan_vqa.py  --load-from ./work_dirs/mcan_vqa.pth --eval-only
```

For more detailed usages, please refer to the [Case 1](1_exist_data_model.md).
