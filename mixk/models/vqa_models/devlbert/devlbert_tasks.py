# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from io import open
from easydict import EasyDict as edict
import torch
import os
from mixk.models.builder import VQA_MODELS
from transformers.modeling_bert import BertConfig
from ..base_model import BaseModel
from .task_utils import compute_score_with_logits
import mixk.utils.distributed_info as comm
import logging
from .devlbert import DeVLBertForVLTasks

logger = logging.getLogger(__name__)


@VQA_MODELS.register_module()
class DEVLBERT(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()

        self.config = config = kwargs['params']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root_path = os.path.dirname(__file__)

        # task_lr = []
        task_ids = []
        for i, task_id in enumerate(config.tasks.split('-')):
            task = 'TASK' + task_id
            cfg = config.TASKS[task]
            name = cfg.name
            task_ids.append(task)
            # task_lr.append(cfg.lr)

        # base_lr = min(task_lr)
        # loss_scale = {}
        # for i, task_id in enumerate(config.tasks.split('-')):
        #     task = 'TASK' + task_id
        # loss_scale[task] = task_lr[i] / base_lr

        train_steps = max([config.TASKS[k]['num_training_steps']
                           for k in task_ids]) // config.gradient_accumulation_steps
        num_labels = max([config.TASKS[k]['num_labels'] for k in task_ids])

        self.task_start_iter = {}
        if len(task_ids) == 1:
            self.task_start_iter[task_ids[0]] = 0
        else:
            for task_id in task_ids:
                self.task_start_iter[task_id] = train_steps - (
                    config.TASKS[task_id]['num_epoch'] * config.TASKS[task_id]['iters_in_epoch'] //
                    config.gradient_accumulation_steps)

        # task_ave_iter_list = sorted(task_ave_iter.values())
        # median_num_iter = task_ave_iter_list[-1]
        # num_train_optimization_steps = (
        #     median_num_iter * \
        #         config.total_epochs // config.gradient_accumulation_steps
        # )

        bertconfig = BertConfig.from_dict(config)

        self.model = DeVLBertForVLTasks.from_pretrained(
            config.from_pretrained,
            config=bertconfig,
            num_labels=num_labels,
        )

        if config.freeze != -1:
            bert_weight_name = json.load(
                open(self.root_path + '/config/' + config.bert_model + '_weight_name.json', 'r'))
            bert_weight_name_filtered = []
            for name in bert_weight_name:
                if 'embeddings' in name:
                    bert_weight_name_filtered.append(name)
                elif 'encoder' in name:
                    layer_num = name.split('.')[2]
                    if int(layer_num) <= config.freeze:
                        bert_weight_name_filtered.append(name)

            for key, value in dict(self.model.named_parameters()).items():
                if key[12:] in bert_weight_name_filtered:
                    value.requires_grad = False

            logger.info('filtered weight')
            logger.info(bert_weight_name_filtered)

        self.lr_reduce_list = [5, 7]
        self.global_step = 0
        self.task_iter_train = {name: None for name in task_ids}
        self.task_count = {name: 0 for name in task_ids}
        self.task_ids = task_ids

        self.is_ema_state = False
        self.bkp_state_dict = None
        self.use_ema = config.TASKS[task_ids[0]]['use_ema']
        self.ema_decay_ratio = config.TASKS[task_ids[0]]['ema_decay_ratio']
        self.ema_state_dict = {}

    def init_ema(self):
        if self.use_ema:
            for param_name, param_tensor in self.model.state_dict().items():
                # we currently store the ema params on GPU
                self.ema_state_dict[param_name] = param_tensor.clone().detach()
        logger.info('init ema state')

    def update_ema(self):
        if self.use_ema:
            self.model.zero_grad()
            for param_name, param_tensor in self.model.state_dict().items():
                assert param_name in self.ema_state_dict
                self.ema_state_dict[param_name] -= (1.0 - self.ema_decay_ratio) * (
                    self.ema_state_dict[param_name] - param_tensor)
            # print('after_train_iter use_ema')

    def load_ema_state_dict(self):
        if self.use_ema and self.is_ema_state is False:
            self.bkp_state_dict = {
                param_name: param_tensor.cpu().detach()
                for param_name, param_tensor in self.model.state_dict().items()
            }
            # load averaged params
            self.model.load_state_dict(self.ema_state_dict)
            self.is_ema_state = True

            logger.info('load ema state dict!')

    def load_bkp_state_dict(self):
        if self.use_ema:
            self.model.load_state_dict(self.bkp_state_dict)
            if self.is_ema_state:
                self.is_ema_state = False

            logger.info('load bkp state dict!')

    def save_checkpoint_ema(self, savePath, epochId):
        # If EMA is used, save averaged model
        if self.use_ema and comm.is_main_process():
            output_ema_state_dict = {}
            for param_name in self.model.state_dict():
                assert param_name in self.ema_state_dict
                if hasattr(self.model, 'module'):
                    output_ema_state_dict[param_name[7:]] = self.ema_state_dict[param_name]  # skip prefix "module."
                else:
                    output_ema_state_dict[param_name] = self.ema_state_dict[param_name]
            output_ema_model_file = os.path.join(savePath, 'pytorch_model_' + str(epochId) + '_ema.bin')
            torch.save(output_ema_state_dict, output_ema_model_file)

            logger.info('Saving ema checkpoint to {}'.format(output_ema_model_file))

    def run_one_time(self, task_id, data):
        params = self.get_image_and_text_features(task_id, data)

        (
            vil_prediction,
            vil_logit,
            vil_binary_prediction,
            vision_prediction,
            vision_logit,
            linguisic_prediction,
            linguisic_logit,
        ) = self.model(
            params.question,
            params.features,
            params.spatials,
            params.segment_ids,
            params.input_mask,
            params.image_mask,
            params.co_attention_mask,
        )

        target = params.target
        batch_size = params.batch_size
        num_options = params.num_options

        cfg_type = self.config.TASKS[task_id]['type']

        if cfg_type == 'VL-classifier':
            batch_score = compute_score_with_logits(vil_prediction, target).sum()
            pred = vil_prediction

        elif cfg_type == 'VL-logit':
            vil_logit = vil_logit.view(batch_size, num_options)
            _, preds = torch.max(vil_logit, 1)
            batch_score = float((preds == target).sum())
            pred = vil_logit

        elif cfg_type == 'V-logit':
            _, select_idx = torch.max(vision_logit, dim=1)
            select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
            batch_score = float(torch.sum(select_target > 0.5))
            pred = vision_logit

        return edict({
            'scores': pred,
            'target': target,
            'batch_score': batch_score,
            'batch_size': batch_size,
        })

    def forward_train(self, data, **kwargs):
        iterId = kwargs['cur_iter']
        # epochId = kwargs['cur_epoch']
        # step = kwargs['inner_iter']

        model_output = {}
        for task_id in self.task_ids:
            if iterId >= self.task_start_iter[task_id]:
                output_dict = self.run_one_time(task_id, data)
                output_dict.batch_score /= output_dict.batch_size

                model_output[task_id] = {
                    'scores': output_dict.scores,
                    'target': output_dict.target,
                    'batch_score': output_dict.batch_score,
                }

        # now only one task
        return model_output[task_id]

    def forward_test(self, data, **kwargs):
        # test now does not support **kwargs
        self.load_ema_state_dict()

        if isinstance(self.task_ids, list):
            task_id = self.task_ids[0]
        else:
            task_id = self.task_ids

        model_output = {}

        output_dict = self.run_one_time(task_id, data)

        model_output[task_id] = {
            'batch_score': output_dict.batch_score,
            'batch_size': output_dict.batch_size,
        }

        # now only one task
        return model_output[task_id]

    def get_image_and_text_features(self, task_id, data):
        batch = tuple(t.cuda(device=self.device, non_blocking=True) for t in data)

        (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
        ) = batch

        num_options = None
        batch_size = features.size(0)
        # cfg_process = self.config.TASKS[task_id]['process']

        if task_id in ['TASK2', 'TASK3', 'TASK6', 'TASK7', 'TASK8']:
            max_num_bbox = features.size(1)
            num_options = question.size(1)
            features = features.unsqueeze(1).expand(batch_size, num_options, max_num_bbox,
                                                    2048).contiguous().view(-1, max_num_bbox, 2048)
            spatials = spatials.unsqueeze(1).expand(batch_size, num_options, max_num_bbox,
                                                    5).contiguous().view(-1, max_num_bbox, 5)
            image_mask = image_mask.unsqueeze(1).expand(batch_size, num_options,
                                                        max_num_bbox).contiguous().view(-1, max_num_bbox)

            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

        elif task_id in ['TASK4']:
            max_num_bbox = features.size(2)
            num_options = question.size(1)
            features = features.expand(batch_size, num_options, max_num_bbox,
                                       2048).contiguous().view(-1, max_num_bbox, 2048)
            spatials = spatials.expand(batch_size, num_options, max_num_bbox, 5).contiguous().view(-1, max_num_bbox, 5)
            image_mask = image_mask.expand(batch_size, num_options, max_num_bbox).contiguous().view(-1, max_num_bbox)

            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

        elif task_id in ['TASK9', 'TASK10']:
            max_num_bbox = features.size(1)
            num_options = question.size(1)
            features = features.view(-1, features.size(2), features.size(3))
            spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
            image_mask = image_mask.view(-1, image_mask.size(2))
            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

        return edict({
            'question': question,
            'features': features,
            'spatials': spatials,
            'segment_ids': segment_ids,
            'input_mask': input_mask,
            'image_mask': image_mask,
            'co_attention_mask': co_attention_mask,
            'target': target,
            'batch_size': batch_size,
            'num_options': num_options,
        })
