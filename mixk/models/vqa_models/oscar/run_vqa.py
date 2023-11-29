# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import logging
import torch

from .modeling.modeling_bert import ImageBertForSequenceClassification
from transformers import BertConfig

from .utils.task_utils import processors
from mixk.models.builder import VQA_MODELS
from ..base_model import BaseModel
import sys

sys.path.insert(0, '.')
logger = logging.getLogger(__name__)


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


@VQA_MODELS.register_module()
class OSCAR(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        args = kwargs['params']
        # Prepare GLUE task
        task_name = args.task_name.lower()
        if task_name not in processors:
            raise ValueError('Task not found: %s' % (task_name))

        num_labels = args.num_labels
        logger.info('Task Name: {}, #Labels: {}'.format(task_name, num_labels))

        self.model_type = args.model_type.lower()
        config = BertConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=task_name)

        # discrete code
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.code_voc = args.code_voc
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.classifier = args.classifier
        config.cls_hidden_scale = args.cls_hidden_scale
        # config.use_img_layernorm = args.use_img_layernorm

        model_class = ImageBertForSequenceClassification
        self.model = model_class.from_pretrained(
            args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

        self.task_name = task_name
        self.img_feature_dim = args.img_feature_dim

    def forward_train(self, data, **kwargs):
        """Train the model."""
        batch = tuple(t.to(self.device) for t in data)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2] if self.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
            'labels': batch[4],
            'img_feats': None if self.img_feature_dim == -1 else batch[5]
        }
        outputs = self.model(**inputs)
        logits = outputs[0]

        batch_score = torch.sum(compute_score_with_logits(logits, batch[4]), 1).sum()
        batch_size = batch[0].size(0)

        model_output = {
            'scores': logits,
            'target': batch[4],
            'batch_score': batch_score,
            'batch_size': batch_size,
        }

        return model_output

    def forward_test(self, data, **kwargs):
        model_output = self.forward_train(data, **kwargs)
        return model_output
