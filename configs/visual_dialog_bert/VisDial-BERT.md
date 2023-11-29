# VisDial*-*BERT

## Introduction

```
@article{visdial_bert
  title={Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline},
  author={Vishvak Murahari and Dhruv Batra and Devi Parikh and Abhishek Das},
  journal={arXiv preprint arXiv:1912.02379},
  year={2019},
}
```

## Training
### To train the base model
```
python tools/run.py --config-file configs/visual_dialog_bert/base_model_on_vqa.py --gpus 4 --load-from xxx/vqa_weights.pth
```
### To finetune the base model with dense annotations
```
python tools/run.py --config-file configs/visual_dialog_bert/ce.py --gpus 4 --load-from xxx/base_model.pth
```

### To finetune the base model with dense annotations and the next sentence prediction(NSP) loss
```
python tools/run.py --config-file configs/visual_dialog_bert/ce+nsp.py --gpus 4 --load-from xxx/base_model.pth
```
## Results and Models

|   Model   | Style   |  NDCG(iMIX) |   NDCG(paper) |
| :------:  | :-----: | :---------: |   :---------: |
| w/cc+vqa  | pytorch | 65.25%      |   64.94%      |
|    CE     | pytorch | 75.78%      |   75.24%      |
|  CE+NSP   | pytorch | 69.84%      |   69.24%      |

**Notes:**

- The NDCG values in the brackets represent those reported in the origin paper of https://github.com/vmurahari3/visdial-bert.
