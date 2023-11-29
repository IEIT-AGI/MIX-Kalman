# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .utils import MultiTaskStopOnPlateau
import json
from io import open
from easydict import EasyDict as edict
import torch
import os
from mixk.models.builder import VQA_MODELS
from transformers.modeling_bert import BertConfig
from ..base_model import BaseModel
from .task_utils import compute_score_with_logits
from .vilbert import VILBertForVLTasks


@VQA_MODELS.register_module()
class VILBERT(BaseModel):

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

        # task_ave_iter = {}
        self.task_stop_controller = {}
        for task_id in task_ids:
            # task_ave_iter[task_id] = int(config.TASKS[task]['num_epoch'] * num_iter *
            #                             config.train_iter_multiplier /
            #                             config.TASKS[task]['num_epoch'])  # config.total_epochs)
            self.task_stop_controller[task_id] = MultiTaskStopOnPlateau(
                mode='max',
                patience=1,
                continue_threshold=0.005,
                cooldown=1,
                threshold=0.001,
            )
        # task_ave_iter_list = sorted(task_ave_iter.values())
        # median_num_iter = task_ave_iter_list[-1]
        # num_train_optimization_steps = (
        #     median_num_iter * \
        #         config.total_epochs // config.gradient_accumulation_steps
        # )
        num_labels = max([config.TASKS[k]['num_labels'] for k in task_ids])

        bertconfig = BertConfig.from_dict(config)

        if bertconfig.visual_target == 0:
            bertconfig.v_config.target_size = 1601
        else:
            bertconfig.v_config.target_size = 2048

        if 'roberta' in config.bert_model:
            bertconfig.model = 'roberta'

        self.model = VILBertForVLTasks.from_pretrained(
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

    def run_one_time(self, task_id, data):
        params = self.get_image_and_text_features(task_id, data)

        (
            vil_prediction,
            vil_prediction_gqa,
            vil_logit,
            vil_binary_prediction,
            vil_tri_prediction,
            vision_prediction,
            vision_logit,
            linguisic_prediction,
            linguisic_logit,
            _,
        ) = self.model(
            params.question,
            params.features,
            params.spatials,
            params.segment_ids,
            params.input_mask,
            params.image_mask,
            params.co_attention_mask,
            params.task_tokens,
        )

        target = params.target
        batch_size = params.batch_size
        multiple_choice_ids = params.multiple_choice_ids
        num_options = params.num_options

        cfg_type = self.config.TASKS[task_id]['type']

        if cfg_type == 'VL-classifier':
            if target is None:
                batch_score = None
            else:
                batch_score = compute_score_with_logits(vil_prediction, target).sum()
            pred = vil_prediction

        elif cfg_type == 'VL-classifier-GQA':
            batch_score = compute_score_with_logits(vil_prediction_gqa, target).sum()
            pred = vil_prediction_gqa

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

        elif cfg_type == 'V-logit-mc':
            vision_logit = vision_logit[:, 101:]
            vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
            vision_logit = vision_logit.unsqueeze(2)
            _, preds = torch.max(vision_logit, dim=1)
            _, target_tmp = torch.max(target, dim=1)
            batch_score = float((preds == target_tmp).sum())
            pred = vision_logit

        elif cfg_type == 'VL-binary-classifier':
            batch_score = compute_score_with_logits(vil_binary_prediction, target).sum()
            pred = vil_binary_prediction

        elif cfg_type == 'VL-tri-classifier':
            batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()
            pred = vil_tri_prediction

        return edict({
            'scores': pred,
            'target': target,
            'batch_score': batch_score,
            'batch_size': batch_size,
        })

    def forward_train(self, data, **kwargs):
        iterId = kwargs['cur_iter']
        epochId = kwargs['cur_epoch']
        step = kwargs['inner_iter']

        # torch.autograd.set_detect_anomaly(True)
        first_task = True
        model_output = {}
        for task_id in self.task_ids:
            is_forward = False
            if (not self.task_stop_controller[task_id].in_stop) or (iterId % self.config.train_iter_gap == 0):
                is_forward = True

            if is_forward:
                output_dict = self.run_one_time(task_id, data)
                output_dict.batch_score /= output_dict.batch_size

                model_output[task_id] = {
                    'scores': output_dict.scores,
                    'target': output_dict.target,
                    'batch_score': output_dict.batch_score,
                }

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # if config.fp16:
                    # lr_this_step = config[learning_rate] * warmup_linear(
                    #     global_step / num_train_optimization_steps,
                    #     config[warmup_proportio]n,
                    # )
                    # for param_group in optimizer.param_groups:
                    #     param_group["lr"] = lr_this_step
                    # if first_task and (
                    #     global_step < warmpu_steps
                    #     or config.lr_scheduler == "warmup_linear"
                    # ):
                    #     warmup_scheduler.step()

                    if first_task:
                        self.global_step += 1
                        first_task = False

        # if "cosine" in config.lr_scheduler and global_step > warmpu_steps:
        #    lr_scheduler.step()
        # if config.lr_scheduler == "automatic":
        #     lr_scheduler.step(sum(val_scores.values()))
        #     logger.info("best average score is %3f" % lr_scheduler.best)
        # elif config.lr_scheduler == "mannul":
        #     lr_scheduler.step()

        if epochId in self.lr_reduce_list:
            for task_id in self.task_ids:
                # reset the task_stop_controller once the lr drop
                self.task_stop_controller[task_id]._reset()

        # now only one task
        return model_output[task_id]

    def forward_test(self, data, **kwargs):
        # test now does not support **kwargs
        if isinstance(self.task_ids, list):
            task_id = self.task_ids[0]
        else:
            task_id = self.task_ids
        # torch.autograd.set_detect_anomaly(True)
        model_output = {}

        output_dict = self.run_one_time(task_id, data)

        if output_dict.target is None:
            model_output[task_id] = {'scores': output_dict.scores}
        else:
            model_output[task_id] = {
                'batch_score': output_dict.batch_score,
                'batch_size': output_dict.batch_size,
            }

        # # update the multi-task scheduler.
        # self.task_stop_controller[task_id].step(
        #     tbLogger.getValScore(task_id))
        # score = tbLogger.showLossVal(task_id, task_stop_controller)

        # now only one task
        return model_output[task_id]

    def get_image_and_text_features(self, task_id, data):

        def convert_tensor(data):
            return data if len(data) else None

        batch = tuple(t.cuda(device=self.device, non_blocking=True) for t in data)
        batch = [convert_tensor(d) for d in batch]

        if task_id == 'TASK4' or task_id == 'TASK17':
            (features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids,
             co_attention_mask, question_id) = (
                 batch)
        else:
            (features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask,
             question_id) = (
                 batch)

        num_options = None
        batch_size = features.size(0)
        cfg_process = self.config.TASKS[task_id]['process']

        if cfg_process in ['dialog']:
            max_num_bbox = features.size(1)
            nround = question.size(1)
            num_options = question.size(2)
            rbatch_size = batch_size * nround
            question = question.view(rbatch_size, question.size(2), question.size(3))
            target = target.view(-1)
            input_mask = input_mask.view(rbatch_size, input_mask.size(2), input_mask.size(3))
            segment_ids = segment_ids.view(rbatch_size, segment_ids.size(2), segment_ids.size(3))
            co_attention_mask = co_attention_mask.view(
                rbatch_size,
                co_attention_mask.size(2),
                co_attention_mask.size(3),
                co_attention_mask.size(4),
            )

            features = (
                features.unsqueeze(1).unsqueeze(1).expand(batch_size, nround, num_options, max_num_bbox,
                                                          2048).contiguous().view(-1, max_num_bbox, 2048))
            spatials = (
                spatials.unsqueeze(1).unsqueeze(1).expand(batch_size, nround, num_options, max_num_bbox,
                                                          5).contiguous().view(-1, max_num_bbox, 5))
            image_mask = (
                image_mask.unsqueeze(1).expand(batch_size, nround, num_options,
                                               max_num_bbox).contiguous().view(-1, max_num_bbox))

            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))
            batch_size = rbatch_size

        elif cfg_process in ['expand']:
            max_num_bbox = features.size(1)
            num_options = question.size(1)
            features = (
                features.unsqueeze(1).expand(batch_size, num_options, max_num_bbox,
                                             2048).contiguous().view(-1, max_num_bbox, 2048))
            spatials = (
                spatials.unsqueeze(1).expand(batch_size, num_options, max_num_bbox,
                                             5).contiguous().view(-1, max_num_bbox, 5))
            image_mask = (
                image_mask.unsqueeze(1).expand(batch_size, num_options,
                                               max_num_bbox).contiguous().view(-1, max_num_bbox))
            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

        elif cfg_process in ['retrieval']:
            max_num_bbox = features.size(1)
            num_options = question.size(1)
            features = features.view(-1, features.size(2), features.size(3))
            spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
            image_mask = image_mask.view(-1, image_mask.size(2))
            question = question.view(-1, question.size(2))
            input_mask = input_mask.view(-1, input_mask.size(2))
            segment_ids = segment_ids.view(-1, segment_ids.size(2))
            co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

        elif cfg_process in ['nlvr']:
            batch_size = features.size(0)
            max_num_bbox = features.size(1)
            num_options = question.size(1)
            features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
            spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
            image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
            question = question.repeat(1, 2)
            question = question.view(batch_size * 2, int(question.size(1) / 2))
            input_mask = input_mask.repeat(1, 2)
            input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
            segment_ids = segment_ids.repeat(1, 2)
            segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
            co_attention_mask = co_attention_mask.view(
                batch_size * 2,
                int(co_attention_mask.size(1) / 2),
                co_attention_mask.size(2),
            )

        task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

        return edict({
            'question': question,
            'features': features,
            'spatials': spatials,
            'segment_ids': segment_ids,
            'input_mask': input_mask,
            'image_mask': image_mask,
            'co_attention_mask': co_attention_mask,
            'task_tokens': task_tokens,
            'target': target,
            'batch_size': batch_size,
            'multiple_choice_ids': multiple_choice_ids if task_id == 'TASK4' or task_id == 'TASK17' else None,
            'num_options': num_options,
        })
