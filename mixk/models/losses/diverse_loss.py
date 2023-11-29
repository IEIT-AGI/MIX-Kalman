import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..builder import LOSSES
from .base_loss import BaseLoss


@LOSSES.register_module()
class DiverseLoss(BaseLoss):
    loss_name = 'diverse_loss'

    def __init__(self):
        super().__init__(loss_name=str(self))

    def forward(self, model_output):
        w_div = 0.125
        predict_anchor = model_output['attnscore_list']
        target_bbox = model_output['input_mask']
        losses = DiverseLoss.compute_loss(predict_anchor, target_bbox)
        losses = w_div * losses
        return losses

    @staticmethod
    def mask_softmax(attn_score, word_mask, tempuature=10., clssep=False, lstm=False):
        if len(attn_score.shape) != 2:
            attn_score = attn_score.squeeze(2).squeeze(2)
        word_mask_cp = word_mask[:, :attn_score.shape[1]].clone()
        score = F.softmax(attn_score * tempuature, dim=1)
        if not clssep:
            for ii in range(word_mask_cp.shape[0]):
                if lstm:
                    word_mask_cp[ii, word_mask_cp[ii, :].sum() - 1] = 0
                else:
                    word_mask_cp[ii, 0] = 0
                    word_mask_cp[ii, word_mask_cp[ii, :].sum()] = 0  # set one to 0 already
        mask_score = score * word_mask_cp.float()
        mask_score = mask_score / (mask_score.sum(1) + 1e-8).view(mask_score.size(0), 1).expand(
            mask_score.size(0), mask_score.size(1))
        return mask_score

    @staticmethod
    def compute_loss(score_list, word_mask, m=-1, coverage_reg=True):
        score_matrix = torch.stack([DiverseLoss.mask_softmax(score, word_mask) for score in score_list],
                                   dim=1)  # (B,Nfilm,N,H,W)
        cov_matrix = torch.bmm(score_matrix, score_matrix.permute(0, 2, 1))  # (BHW,Nfilm,Nfilm)
        id_matrix = Variable(torch.eye(cov_matrix.shape[1]).unsqueeze(0).repeat(cov_matrix.shape[0], 1, 1).cuda())
        if m == -1.:
            div_reg = torch.sum(((cov_matrix * (1 - id_matrix))**2).view(-1)) / cov_matrix.shape[0]
        else:
            div_reg = torch.sum(((cov_matrix - m * id_matrix)**2).view(-1)) / cov_matrix.shape[0]
        if coverage_reg:
            word_mask_cp = word_mask.clone()
            for ii in range(word_mask_cp.shape[0]):
                word_mask_cp[ii, 0] = 0
                word_mask_cp[ii, word_mask_cp[ii, :].sum()] = 0  # set one to 0 already
            cover_matrix = 1. - torch.clamp(torch.sum(score_matrix, dim=1, keepdim=False), min=0., max=1.)
            cover_reg = torch.sum((cover_matrix * word_mask_cp.float()).view(-1)) / cov_matrix.shape[0]
            div_reg += cover_reg
        return div_reg
