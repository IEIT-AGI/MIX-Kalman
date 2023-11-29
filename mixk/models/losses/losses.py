import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss
from torch.nn import SmoothL1Loss as TorchSmoothL1Loss

from ..builder import LOSSES, build_loss
from .base_loss import BaseLoss
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Dict


@LOSSES.register_module()
class TripleLogitBinaryCrossEntropy(BaseLoss):
    """This is used for Three-branch fusion only.

    We predict scores and compute cross entropy loss for each of branches.
    """

    loss_name = 'triple_logit_binary_cross_entropy_loss'

    def __init__(self):
        super().__init__()

    def forward(self, model_output):
        """Calculates and returns the binary cross entropy for logits
            Args:
                sample_list (SampleList): SampleList containing `targets` attribute.
                model_output (Dict): Model output containing `scores` attribute.
            Returns:
                torch.FloatTensor: Float value for loss.
            """

        scores, target = model_output['scores'], model_output['target']

        if scores.dim() == 3:
            loss = (
                F.binary_cross_entropy_with_logits(scores[:, 0], target, reduction='mean') +
                F.binary_cross_entropy_with_logits(scores[:, 1], target, reduction='mean') +
                F.binary_cross_entropy_with_logits(scores[:, 2], target, reduction='mean'))
        else:
            loss = F.binary_cross_entropy_with_logits(scores, target, reduction='mean')

        return loss * target.size(-1)


@LOSSES.register_module()
class BinaryCrossEntropyWithLogits(BaseLoss):
    loss_name = 'binary_cross_entropy_with_logits_loss'

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, model_output):
        predict_scores, target = model_output['scores'], model_output['target']
        return self.loss_fn(predict_scores, target)


@LOSSES.register_module()
class CrossEntropyLoss(BaseLoss):
    loss_name = 'cross_entropy_loss'

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, model_output):
        predict_scores, target = model_output['scores'], model_output['target']
        return self.loss_fn(predict_scores, target)


@LOSSES.register_module()
class OBJCrossEntropyLoss(BaseLoss):
    loss_name = 'obj_cross_entropy_loss'

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, model_output):
        predict_scores, target = model_output['obj_scores'], model_output['obj_target']
        return self.loss_fn(predict_scores, target)


@LOSSES.register_module()
class LogitBinaryCrossEntropy(BaseLoss):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """

    loss_name = 'logit_binary_cross_entropy_loss'

    def __init__(self):
        super().__init__()

    def forward(self, model_output):
        """Calculates and returns the binary cross entropy for logits.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.
        """

        scores, targets = model_output['scores'], model_output['target']
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction='mean')

        return loss * targets.size(1)


@LOSSES.register_module()
class CaptionCrossEntropyLoss(BaseLoss):
    loss_name = 'caption_cross_entropy_loss'

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output['scores']
        targets = sample_list['targets']

        # If no captions(test dataset) then assume decode length to be uniform
        if hasattr(sample_list, 'caption_len'):
            caption_lengths, _ = sample_list.caption_len.sort(dim=0, descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
        else:
            decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= '1.1':
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = F.cross_entropy(scores, targets)

        return loss


@LOSSES.register_module()
class M4CDecodingBCEWithMaskLoss(BaseLoss):
    loss_name = 'M4CDecodingBCEWithMask_loss'

    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.0])

    def forward(self, model_output):
        scores = model_output['scores']
        targets = model_output['target']
        loss_mask = model_output['train_loss_mask']
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits(scores, targets, reduction='none')
        losses *= loss_mask.unsqueeze(-1)

        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss


@LOSSES.register_module()
class LXMERTPreTrainLossV0(BaseLoss):
    loss_name = 'lxmert_pretrain_v0_loss'

    def __init__(self, visual_losses, visual_loss_config, vocab_size, num_answers):
        super().__init__()
        self.loss_fct_cls = TorchCrossEntropyLoss(ignore_index=-1)
        self.loss_fcts_feat = {
            'l2': TorchSmoothL1Loss(reduction='none'),
            'ce': TorchCrossEntropyLoss(ignore_index=-1, reduction='none')
        }
        self.visual_losses = visual_losses.split(',')
        self.visual_loss_config = visual_loss_config
        self.vocab_size = vocab_size
        self.num_answers = num_answers

    def forward(self, model_output):
        scores = model_output['scores']
        target = model_output['target']
        lang_prediction_scores = scores['lang_prediction_scores']
        cross_relationship_score = scores['cross_relationship_score']
        visn_prediction_scores_dict = scores['visn_prediction_scores_dict']
        answer_score = scores['answer_score']
        masked_lm_labels = target['masked_lm_labels']
        matched_label = target['matched_label']
        obj_labels = target['obj_labels']
        ans = target['ans']

        total_loss = 0.
        losses = ()

        masked_lm_loss = self.loss_fct_cls(lang_prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        total_loss += masked_lm_loss
        losses += (masked_lm_loss.detach(), )

        matched_loss = self.loss_fct_cls(cross_relationship_score.view(-1, 2), matched_label.view(-1))
        total_loss += matched_loss
        losses += (matched_loss.detach(), )

        total_visn_loss = 0.
        for key in self.visual_losses:
            label, mask_conf = obj_labels[key]
            output_dim, loss_fct_name, label_shape, weight = self.visual_loss_config[key]
            visn_loss_fct = self.loss_fcts_feat[loss_fct_name]
            visn_prediction_scores = visn_prediction_scores_dict[key]
            visn_loss = visn_loss_fct(
                visn_prediction_scores.view(-1, output_dim),
                label.view(*label_shape),
            )
            if visn_loss.dim() > 1:  # Regression Losses
                visn_loss = visn_loss.mean(1)
            visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
            total_visn_loss += visn_loss
            losses += (visn_loss.detach(), )
        total_loss += total_visn_loss

        answer_loss = self.loss_fct_cls(answer_score.view(-1, self.num_answers), ans.view(-1))

        total_loss += answer_loss
        losses += (answer_loss.detach(), )

        return total_loss  # , torch.stack(losses).unsqueeze(0), answer_score.detach()


@LOSSES.register_module()
class VILBERTMutilLoss(BaseLoss):
    loss_name = 'vilbert_mutil_loss'

    def __init__(self, task_cfg):
        super().__init__()
        self.LossMap = {
            'BCEWithLogitLoss': nn.BCEWithLogitsLoss(reduction='mean'),
            'CrossEntropyLoss': nn.CrossEntropyLoss(),
        }
        self.task_ids = []
        self.loss_scale = {}
        self.task_cfg = task_cfg
        self.task_losses = self.load_losses()

    def load_losses(self):
        losses = {}
        task_types = []

        for i, task_id in enumerate(self.task_cfg['tasks'].split('-')):
            task = 'TASK' + task_id
            cfg = self.task_cfg.TASKS[task]
            model_type = cfg.type
            if model_type not in task_types:
                task_types.append(model_type)
            losses[task] = self.LossMap[cfg.loss]
            self.loss_scale[task] = cfg.loss_scale
            self.task_ids.append(task)

        return losses

    def forward(self, model_output):
        for task_id in self.task_ids:
            # only one task now
            pred = model_output['scores']  # model_output[task_id]['scores']
            target = model_output['target']  # model_output[task_id]['target']

            # for different task, we use different output to calculate the loss.
            loss = self.task_losses[task_id](pred, target)
            task_type = self.task_cfg.TASKS[task_id]['type']
            if task_type in ['VL-classifier', 'VL-classifier-GQA', 'V-logit', 'V-logit-mc']:
                loss = loss.mean() * target.size(1)
            elif task_type in ['VL-binary-classifier', 'VL-tri-classifier']:
                loss = loss.mean()

            loss = loss * self.loss_scale[task_id]

        return loss


@LOSSES.register_module()
class BCEWithLogitsLoss(BaseLoss):
    """Returns .

    Attention:
        `Key`: logit_bce
    """
    loss_name = 'bce_with_logits_loss'

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean', **params)

    def forward(self, model_output):
        """Calculates and returns the binary cross entropy for logits.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.
        """

        scores, targets = model_output['scores'], model_output['target']
        return self.loss_fn(scores, targets)


@LOSSES.register_module()
class OSCARLoss(BaseLoss):
    loss_name = 'oscar_mutil_loss'

    def __init__(self, cfg):
        super().__init__()
        self.loss_type = cfg.loss_type
        self.num_labels = cfg.num_labels

    def instance_bce_with_logits(self, logits, labels, reduction='mean'):
        assert logits.dim() == 2
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
        if reduction == 'mean':
            loss *= labels.size(1)
        return loss

    def forward(self, model_output):
        logits = model_output['scores']
        labels = model_output['target']

        if labels is not None:
            if self.num_labels == 1:  # doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'kl':
                    loss_fct = nn.KLDivLoss(reduction='batchmean')
                    log_softmax = nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce':  # [VQA]
                    loss = self.instance_bce_with_logits(logits, labels)
                else:  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        return loss


@LOSSES.register_module()
class OSCARBertCaptioningLoss(nn.Module):
    loss_name = 'oscar_bert_captioning_loss'

    def __init__(self, cfg):
        super().__init__()
        self.label_smoothing = getattr(cfg, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(cfg, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(cfg, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')
        self.iter = 0

    def __str__(self):
        return 'oscar_bert_captioning_loss'

    def forward(self, model_output):
        logits = model_output['scores']
        target = model_output['target']

        self.iter += 1
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = torch.topk(loss, k=int(loss.shape[0] * (1 - self.drop_worst_ratio)), largest=False)

        loss = loss.mean()

        return loss


@LOSSES.register_module()
class VisualDialogBertLoss(BaseLoss):
    loss_name = 'visual_dialog_bert_loss'

    def __init__(self, MLM_loss: Dict, NSP_loss: Dict, MIR_loss: Dict, predict_feature: bool = False):
        super().__init__()

        self.masked_lm_loss_coeff = MLM_loss.pop('weight_coeff')
        self.masked_lm_loss = build_loss(cfg=MLM_loss)

        self.masked_img_loss_coeff = MIR_loss.pop('weight_coeff')
        self.masked_img_loss = build_loss(cfg=MIR_loss)

        self.next_sentence_pred_loss_coeff = NSP_loss.pop('weight_coeff')
        self.next_sentence_pred_loss = build_loss(cfg=NSP_loss)

        self.predict_feature = predict_feature

    def forward(self, model_output):
        visual_predict_scores = model_output['visual_predict_scores']
        image_target = model_output['image_target']
        masked_img_loss = self.masked_img_loss_coeff * self.calcuate_img_loss(visual_predict_scores, image_target)

        text_predict_scores = model_output['text_predict_scores']
        masked_lm_labels = model_output['masked_lm_labels']
        masked_lm_loss = self.masked_lm_loss_coeff * self.calcuate_text_loss(text_predict_scores, masked_lm_labels)

        seq_relationship_scores = model_output['seq_relationship_scores']
        next_sentence_label = model_output['next_sentence_label']
        nsp_loss = self.next_sentence_pred_loss_coeff * self.calcuate_nsp_loss(seq_relationship_scores,
                                                                               next_sentence_label)

        output_loss = masked_lm_loss + masked_lm_loss + nsp_loss

        return {
            str(self): output_loss,
            'masked_img_loss': masked_img_loss,
            'nsp_loss': nsp_loss,
            'masked_lm_loss': masked_lm_loss
        }

    def calcuate_img_loss(self, prediction, target):
        # model_output['scores'], model_output['target']
        if self.predict_feature:
            img_loss = self.masked_img_loss(model_output={'scores': prediction, 'target': target})
            max_v = max(torch.sum((target == 1).unsqueeze(2).expand_as(img_loss)), 1)
        else:
            img_loss = self.masked_img_loss(model_output={'scores': F.log_softmax(prediction, dim=2), 'target': target})
            max_v = max(torch.sum((target == 1)), 0)

        sum_v = torch.sum(img_loss * (target == 1))
        return sum_v / max_v

    def calcuate_text_loss(self, prediction, target):
        return self.masked_lm_loss(model_output={'scores': prediction, 'target': target})

    def calcuate_nsp_loss(self, prediction, target):
        return self.next_sentence_pred_loss(model_output={'scores': prediction, 'target': target})


@LOSSES.register_module()
class KLDivLoss(BaseLoss):
    loss_name = 'KLDiv_loss'

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.KLDivLoss(**params)

    def forward(self, model_output):
        predict_scores, target = model_output['scores'], model_output['target']
        return self.loss_fn(predict_scores, target)


@LOSSES.register_module()
class VisualDialogBertDenseLoss(BaseLoss):
    loss_name = 'visual_dialog_bert_dense_loss'

    def __init__(self, NSP_loss: Dict, KLDiv_loss: Dict, MLM_loss: Dict, MIR_loss: Dict):
        super().__init__()

        self.nsp_loss_coeff = NSP_loss.pop('weight_coeff')  # next sentence prediction loss
        self.nsp_loss_fun = build_loss(cfg=NSP_loss)

        self.kldiv_loss_coef = KLDiv_loss.pop('weight_coeff')
        self.kldiv_loss_fun = build_loss(cfg=KLDiv_loss)

        self.mlm_loss_coeff = MLM_loss.pop('weight_coeff')  # mask language modeling loss
        self.mlm_loss_fun = build_loss(cfg=MLM_loss)

        self.mir_loss_coeff = MIR_loss.pop('weight_coeff')  # masked image region loss
        self.mir_loss_fun = build_loss(cfg=MIR_loss)

        self.predict_feature = False

    def forward(self, model_output):
        nsp_scores = model_output['seq_relationship_scores']
        nsp_scores = nsp_scores.view(-1, nsp_scores.shape[0], nsp_scores.shape[1])
        next_sentence_label = model_output['next_sentence_label']
        nsp_loss = self.nsp_loss_fun({
            'scores': nsp_scores.view(-1, 2),
            'target': next_sentence_label.view(-1)
        }) * self.nsp_loss_coeff

        nsp_scores = nsp_scores[:, :, 0]
        gt_relevance = model_output['gt_relevance']
        kldiv_loss = self.kldiv_loss_fun({
            'scores': F.log_softmax(nsp_scores, dim=1),
            'target': F.softmax(gt_relevance, dim=1)
        }) * self.kldiv_loss_coef

        # torch runtimeError
        visual_predict_scores = model_output['visual_predict_scores']
        image_target = model_output['image_target']
        masked_img_loss = self.mir_loss_coeff * self.calcuate_img_loss(visual_predict_scores, image_target)

        text_predict_scores = model_output['text_predict_scores']
        masked_lm_labels = model_output['masked_lm_labels']
        masked_lm_loss = self.mlm_loss_coeff * self.calcuate_text_loss(text_predict_scores, masked_lm_labels)

        loss = kldiv_loss + nsp_loss + masked_lm_loss + masked_img_loss

        return {
            str(self): loss,
            'kldiv_loss': kldiv_loss,
            'nsp_loss': nsp_loss,
            'masked_img_loss': masked_img_loss,
            'masked_lm_loss': masked_lm_loss
        }

    def calcuate_img_loss(self, prediction, target):
        # model_output['scores'], model_output['target']
        if self.predict_feature:
            img_loss = self.mir_loss_fun(model_output={'scores': prediction, 'target': target})
            max_v = max(torch.sum((target == 1).unsqueeze(2).expand_as(img_loss)), 1)
        else:
            img_loss = self.mir_loss_fun(model_output={'scores': F.log_softmax(prediction, dim=2), 'target': target})
            max_v = max(torch.sum((target == 1)), 0)

        sum_v = torch.sum(img_loss * (target == 1))
        return sum_v / max_v

    def calcuate_text_loss(self, prediction, target):
        return self.mlm_loss_fun(model_output={'scores': prediction, 'target': target})


@LOSSES.register_module()
class PretrainLoss(BaseLoss):
    loss_name = 'pretraining_loss'

    def __init__(self):
        pass

    def forward(self, model_output):
        output = {}
        for k, v in model_output.losses.items():
            output[k] = v

        output[str(self)] = model_output.loss

        return output
