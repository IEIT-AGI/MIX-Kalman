from ..builder import VQA_MODELS, build_backbone, build_encoder, build_head
import torch.nn as nn
import torch
import torch.nn.functional as F
from mixk.models.backbones.lcgn_backbone import Linear, apply_mask1d
from .base_model import BaseModel
from typing import Dict
from torch import Tensor


@VQA_MODELS.register_module()
class LCGN(BaseModel):

    def __init__(self, encoder: 'LCGNEncoder', backbone: 'LCGN_BACKBONE', head: 'LCGNClassiferHead') -> None:
        """The initialization of LCGN model.

        Args:
            encoder: the encoder of LCGN, which is used to encode the questions
            backbone: the backbon of LCGN, which extracts the context feature by message passing
            head: the classifier head of LCGN,
        """
        super().__init__()

        self.encoder_model = build_encoder(encoder)
        self.backbone = build_backbone(backbone)
        self.single_hop = SingleHop(self.backbone.CTX_DIM, self.encoder_model.ENC_DIM)
        self.head = build_head(head)  # 包括 classification head， generation head

    def forward_train(self, data: Dict, **kwargs) -> Dict:
        """the forward computation of LCGN.

        Args:
            data: A dict containing the inputs, using data.keys() to get the specific content

        Returns:
            Dict: the output of the ground truth answers and predicted answers, formed to a dictionary
        """
        questionIndices = data['input_ids'].cuda()
        questionLengths = data['questionLengths']
        images = torch.cat((data['feature'], data['bbox'].repeat(1, 1, 16)), dim=-1).cuda()
        # batchSize = data['image_dim'].shape[0]
        imagesObjectNum = data['image_dim'].cuda()

        questionCntxWords, vecQuestions = self.encoder_model(questionIndices, questionLengths)

        # LCGN
        x_out = self.backbone(
            images=images,
            q_encoding=vecQuestions,
            lstm_outputs=questionCntxWords,
            q_length=questionLengths,
            entity_num=imagesObjectNum)

        # Single-Hop
        x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
        model_output = {'scores': self.head.forward(x_att, vecQuestions), 'target': data['answers_scores'].cuda()}
        return model_output

    def forward_test(self, data: Dict) -> Dict:
        """call forward_train to complete test task.

        Args:
            data: A dict containing the inputs, using data.keys() to get the specific content

        Returns:
            Dict: the output of the ground truth answers and predicted answers, formed to a dictionary
        """
        model_output = self.forward_train(data)
        return model_output


class SingleHop(nn.Module):

    def __init__(self, CTX_DIM: int, ENC_DIM: int) -> None:
        """initialzation of SingleHop.

        Args:
            CTX_DIM: context dimension
            ENC_DIM: encoder dimension
        """

        super().__init__()
        self.CTX_DIM = CTX_DIM
        self.ENC_DIM = ENC_DIM

        self.proj_q = Linear(self.ENC_DIM, self.CTX_DIM)
        self.inter2att = Linear(self.CTX_DIM, 1)

    def forward(self, kb: Tensor, vecQuestions: Tensor, imagesObjectNum: Tensor) -> Tensor:
        """forward computation of SingleHop.

        Args:
            kb: the output feature of backbone, is x^{out}_{i} in paper
            vecQuestions: question vector, is q in paper
            imagesObjectNum: the number of objects in every sample

        Returns:
            Tensor: the weighted output of x^{out}_{i} in paper
        """
        proj_q = self.proj_q(vecQuestions)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        raw_att = self.inter2att(interactions).squeeze(-1)  # (batch_size, 36)
        raw_att = apply_mask1d(raw_att, imagesObjectNum)
        att = F.softmax(raw_att, dim=-1)

        x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
        return x_att
