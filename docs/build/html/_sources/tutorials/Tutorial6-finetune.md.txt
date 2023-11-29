# Tutorial 6: Finetuning Models

There are two steps to finetune a model on a new dataset.

- Add support for the new dataset following [Tutorial2](Tutorial2-customize_dataset.md).
- Modify the configs as will be discussed in this tutorial.

Take the finetuning process on VQA Dataset as an example, the users need to modify five parts in the config.

## Modify dataset

You need to prepare the dataset and write the configs about dataset. MIX-Kalman already support VQA, GQA, OCR-VQA, RefCOCO, VCR, VisDial, and etc.

For more details about customize new dataset, please refer to the [Tutorial2](Tutorial2-customize_dataset.md).

## Modify head

The key point is as follows:

- Modify the output dimension in the model config file according to your class numbers of the new datasets (e.g. LXMERT and VQA task,  modify the `num_labels` parameter to 3129).

  ```
  model = dict(
      type='LXMERT',
      params=dict(
          num_labels=3129,  # set the output dimension same with the dataset class
      ...
      ))
  ```

- the weights of the pre-trained models are mostly reused except the final prediction head.

- you can also modify the head type or stucture according to your task.

## Modify training schedule

The fine tuning hyper parameters vary from the default schedule. It usually requires smaller learning rate and less training epochs.

```python
optimizer = dict(
    type='BertAdam',
    lr=5e-5,
    weight_decay=0.01,
    eps=1e-6,
    betas=[0.9, 0.999],
    max_grad_norm=-1,
    training_encoder_lr_multiply=1,
)
optimizer_config = dict(grad_clip=dict(max_norm=5))
'''
fp16 = dict(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)
'''

lr_config = dict(
    warmup=0.1,
    warmup_method='warmup_linear',
    # max_iters=55472,  # ceil(totoal 443753 / batch size 32) * epoch size  datasets: train
    max_iters=79012,  # floor(totoal 632117 / batch size 32) * epoch size  datasets: train, nominival
    policy='BertWarmupLinearLR')

# by_iter = True
total_epochs = 4
```

## Set base configs

Set the configuration path of the model, dataset, schedule, and runtime in the `configs` directory. (e.g. [LXMERT model and VQA task](https://github.com/IEIT-AGI/MIX-Kalman/tree/master/configs/lxmert/lxmert_vqa.py)).

```python
_base_ = [
    '../_base_/models/lxmert/lxmert_vqa_config.py',
    '../_base_/datasets/lxmert/lxmert_vqa.py',
    '../_base_/default_runtime.py',
]
```

## Use pre-trained model

You could download the model weights before training to avoid waiting time. Model pth file is available [here](https://mega.nz/file/OW5GEIxb#TeXyG2OhV8ZoQ2ESGZOyhONlK0B9p0qwG4bBSkyIX0c).

To use the pre-trained model, you can set the pre-trained  model path in the `--load-from` in the script.

```python
--load-from '/home/datasets/mix_data/mixk/data/models/model_LXRT.pth'
```

or you can set the `load_from` in the `configs/_base_/default_runtime.py`

```python
--load-from '/home/datasets/mix_data/mixk/data/models/model_LXRT.pth'
```
