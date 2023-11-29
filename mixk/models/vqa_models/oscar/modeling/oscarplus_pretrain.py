from mixk.models.builder import VQA_MODELS
from mixk.models.vqa_models.base_model import BaseModel
from copy import deepcopy
from pytorch_transformers import BertConfig
from .modeling_bert import BertImgForPreTraining

from mixk.utils.registry import Registry
from collections import OrderedDict
import torch

BERT = Registry('bert')
BERT.register_module('BertConfig', BertConfig)
BERT.register_module('BertImgForPreTraining', BertImgForPreTraining)


@VQA_MODELS.register_module()
class OscarPreTraining(BaseModel):

    def __init__(self, bert_cfg, pretrained_cfg):
        super().__init__()
        self.bert_cfg = bert_cfg
        self.pretrained_cfg = pretrained_cfg

        config = self.build_bert_cfg()
        self.pretrain_model = self.build_pretrain_model(config)

    def build_bert_cfg(self):

        def num_contrast_classes(texta_false_prob, use_b):
            if texta_false_prob < 0.5 and (texta_false_prob > 0 or not use_b):
                return 3
            else:
                return 2

        bert_cfg = deepcopy(self.bert_cfg)
        obj_type = bert_cfg.pop('type')
        obj_class = BERT.get(obj_type)

        method_cfg = bert_cfg.pop('run_method')
        run_fun = method_cfg.pop('type')
        cfg = getattr(obj_class, run_fun)(**method_cfg)

        texta_false_prob = bert_cfg.pop('texta_false_prob')
        use_b = bert_cfg.pop('use_b')

        cfg.num_contrast_classes = num_contrast_classes(texta_false_prob, use_b)

        for k, v in bert_cfg.items():
            setattr(cfg, k, v)

        return cfg

    def build_pretrain_model(self, config):
        pretrained_cfg = deepcopy(self.pretrained_cfg)
        obj_type = pretrained_cfg.pop('type')
        obj_class = BERT.get(obj_type)

        method_cfg = pretrained_cfg.pop('run_method')
        run_fun = method_cfg.pop('type')

        method_cfg.from_tf = bool('.ckpt' in method_cfg.pretrained_model_name_or_path)
        method_cfg.config = config

        method_params = OrderedDict()
        order_keys = ['pretrained_model_name_or_path', 'from_tf', 'config', 'cache_dir']
        for k in order_keys:
            method_params[k] = method_cfg[k]

        return getattr(obj_class, run_fun)(**method_params)

    def forward_train(self, mini_batch, **kwargs):
        images, input_ids, input_mask, segment_ids, lm_label_ids, is_next, _, _ = self.data_process(mini_batch)
        image_features = torch.stack(images).to('cuda', non_blocking=True)
        outputs = self.pretrain_model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            masked_lm_labels=lm_label_ids,
            next_sentence_label=is_next,
            img_feats=image_features)
        return outputs

    def forward_test(self, data, **kwargs):
        pass

    def data_process(self, mini_batch):
        device = 'cuda'
        images, targets, qa_inds = mini_batch[0], mini_batch[1], mini_batch[2]
        targets_transposed = list(zip(*targets))
        input_ids = torch.stack(targets_transposed[0]).to(device, non_blocking=True)
        input_mask = torch.stack(targets_transposed[1]).to(device, non_blocking=True)
        segment_ids = torch.stack(targets_transposed[2]).to(device, non_blocking=True)
        lm_label_ids = torch.stack(targets_transposed[3]).to(device, non_blocking=True)
        is_next = torch.stack(targets_transposed[4]).to(device, non_blocking=True)
        is_img_match = torch.stack(targets_transposed[5]).to(device, non_blocking=True)

        return images, input_ids, input_mask, segment_ids, lm_label_ids, is_next, qa_inds, is_img_match
