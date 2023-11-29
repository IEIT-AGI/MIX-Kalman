
# MIX·Kalman

## Introduction

English | [简体中文](README_zh-CN.md)

Multimodal Artificial Intelligence Framework (MIX·Kalman) is an open source multi-modal model building toolbox. This framework is based on the out-of-the-box design concept. It is compatible with rich multi-modal tasks, models and datasets. It is scalable, ease to use and in high performance.

The master branch works based on PyTorch.

![demo_image](resources/0.png)

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.1.0 is our first release. MIX·Kalman v0.1.0 supports mainstream multi-modal datasets, models and mixed precision training. And it supports distribute training across multiple GPUs and multiple nodes.

MIX·Kalman's subsequent version will optimize the framework further. We will add more dual-stream and single-stream pre-training models, add more data process methods such as mask, back translation and unsupervised data enhancement, and support launch multiple jobs for training on a single machine simultaneously.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/getstart/model_zoo.md).

All supported models and tasks are shown in the table below.

Supported backbones:

| task              | LXMERT | UNITER | ViLBERT | DeVLBert | Oscar | VinVL | MCAN | LCGN | HGL  | R2C  | VisDial-BERT |
| ----------------- | ------ | ------ | ------- | -------- | ----- | ----- | ---- | ---- | ---- | ---- | ------------ |
| VQA               | √      | √      | √       | √        | √     | √     | √    |      |      |      |              |
| GQA               | √      |        | √       |          | √     | √     |      | √    |      |      |              |
| NLVR              | √      | √      |         |          | √     | √     |      |      |      |      |              |
| VQA_large         |        |        |         |          | √     |       |      |      |      |      |              |
| NLVR_large        |        |        |         |          | √     | √     |      |      |      |      |              |
| GussWhatPointing  |        |        | √       |          |       |       |      |      |      |      |              |
| VisualEntailment  |        | √      | √       |          |       |       |      |      |      |      |              |
| GussWhat          |        |        | √       |          |       |       |      |      |      |      |              |
| VCR_QAR           |        |        |         | √        |       |       |      |      | √    | √    |              |
| VCR_QA            |        |        |         | √        |       |       |      |      | √    | √    |              |
| Visual7w          |        |        | √       |          |       |       |      |      |      |      |              |
| RetrivalFlickr30k |        |        | √       |          |       |       |      |      |      |      |              |
| GenomeQA          |        |        | √       |          |       |       |      |      |      |      |              |
| Retrivalcoco      |        |        | √       |          |       |       |      |      |      |      |              |
| refcocog          |        |        | √       |          |       |       |      |      |      |      |              |
| refcoco           |        |        | √       |          |       |       |      |      |      |      |              |
| refcoco+          |        |        | √       | √        |       |       |      |      |      |      |              |
| VisDial           |        |        |         |          |       |       |      |      |      |      | √            |

## Installation

Please refer to [get_started.md](docs/getstart/get_started.md) for installation.

## Getting Started

Please see [quickrun](docs/Quickrun/1_exist_data_model.md) for the basic usage of MIX·Kalman and visual interface for inference.
We provide basic introduction of MIX·Kalman core module [engine](docs/tutorials/Tutorial-engine.md), full guidance for [configuration](docs/tutorials/Tutorial1-config.md), and all the [results and model](docs/getstart/model_zoo.md).
There are also tutorials for [finetuning models](docs/tutorials/Tutorial6-finetune.md), [adding new dataset](docs/tutorials/Tutorial2-customize_dataset.md), [customizing models](docs/tutorials/Tutorial3-customize_models.md), [customizing runtime settings](docs/tutorials/Tutorial4-customize_Schedule_and_Runtime_Settings.md) and [useful tools](docs/log_visualization.md).

## Contributing

We appreciate all contributions to improve MIX·Kalman. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MIX·Kalman is an open source project that is contributed by researchers and engineers from IEIT. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@misc{fan2021MIX·Kalman,
  author =       {Baoyu Fan, Liang Jin, Runze Zhang, Xiaochuan Li, Cong Xu, Hongzhi Shi, Jian Zhao, Yinyin Chao, Yingjie Zhang, Binqiang Wang, Zhenhua Guo, Yaqian Zhao, Rengang Li},
  title =        {MIX·Kalman: A multimodal framework for vision and language research},
  howpublished = {[MIX-Kalman]{https://github.com/IEIT-AGI/MIX-Kalman}},
  year =         {2021}
}
```
