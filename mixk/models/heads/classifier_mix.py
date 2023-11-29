import math
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from mixk.models.backbones.lcgn_backbone import Linear
from mixk.models.combine_layers import ReLUWithWeightNormFC
from ..builder import HEADS

from torch import Tensor


def gelu(x):
    """Implementation of the gelu activation function. For information: OpenAI
    GPT's gelu is slightly different (and gives slightly different results):

    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):

    def forward(self, input_):
        output = gelu(input_)
        return output


@HEADS.register_module()
class ClassifierHead(nn.Module, metaclass=ABCMeta):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


@HEADS.register_module()
class BertClassifierHead(ClassifierHead):

    def __init(self, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config is None:
            from transformers.configuration_bert import BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')
            assert config.hidden_size == self.in_dim
        from transformers.modeling_bert import BertPredictionHeadTransform
        self.module = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            BertPredictionHeadTransform(config),
            nn.Linear(self.in_dim, self.out_dim),
        )

    def forward(self, *args, **kwargs):
        pass


@HEADS.register_module()
class MLPClassiferHead(ClassifierHead):

    def __init__(self, hidden_dim=None, num_layers=0, dropout=0.5, hidden_act='relu', batch_norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from mixk.utils.modeling import ACT2FN

        activation = ACT2FN[hidden_act]
        self.layers = nn.ModuleList()

        if hidden_dim is None:
            hidden_dim = self.in_dim

        for _ in range(num_layers):
            self.layers.append(nn.Linear(self.in_dim, hidden_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(activation())
            self.layers.append(nn.Dropout(dropout))
            self.in_dim = hidden_dim

        self.layers.append(nn.Linear(self.in_dim, self.out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@HEADS.register_module()
class LogitClassifierHead(ClassifierHead):

    def __init__(self, *args, **kwargs):
        text_non_linear_dim = kwargs.pop('text_hidden_dim')
        image_non_linear_dim = kwargs.pop('img_hidden_dim')
        pretrained_image_data, pretrained_text_data = None, None
        if 'pretrained_image' in kwargs:
            pretrained_image_data = kwargs.pop('pretrained_image')
        if 'pretrained_text' in kwargs:
            pretrained_text_data = kwargs.pop('pretrained_text')

        super().__init__(*args, **kwargs)

        input_dim = self.in_dim
        num_ans_candidates = self.out_dim
        self.f_o_text = ReLUWithWeightNormFC(input_dim, text_non_linear_dim)
        self.f_o_image = ReLUWithWeightNormFC(input_dim, image_non_linear_dim)
        self.linear_text = nn.Linear(text_non_linear_dim, num_ans_candidates)
        self.linear_image = nn.Linear(image_non_linear_dim, num_ans_candidates)

        if pretrained_image_data and pretrained_text_data:
            self.linear_text.weight.data.copy_(torch.from_numpy(pretrained_text_data))
            self.linear_image.weight.data.copy_(torch.from_numpy(pretrained_image_data))

    def forward(self, joint_embedding):
        text_val = self.linear_text(self.f_o_text(joint_embedding))
        image_val = self.linear_image(self.f_o_image(joint_embedding))
        logit_value = text_val + image_val

        return logit_value


@HEADS.register_module()
class LCGNClassiferHead(ClassifierHead):

    def __init__(self, OUT_QUESTION_MUL: bool, CMD_DIM: int, outputDropout: float, *args, **kwargs) -> None:
        """initialization of LCGNClassiferHead.

        Args:
          OUT_QUESTION_MUL: bool to identify if do the multiplication opearation based on features
          CMD_DIM: command vector's dimension
          outputDropout: dropout rate in output layer
          *args: optional
          **kwargs: optional
        """
        super().__init__(*args, **kwargs)
        self.OUT_QUESTION_MUL = OUT_QUESTION_MUL
        self.outQuestion = Linear(CMD_DIM, CMD_DIM)
        self.in_dim = 3 * self.in_dim if OUT_QUESTION_MUL else 2 * self.in_dim
        self.classifier_layer = nn.Sequential(
            nn.Dropout(1 - outputDropout), Linear(self.in_dim, CMD_DIM), nn.ELU(), nn.Dropout(1 - outputDropout),
            Linear(CMD_DIM, self.out_dim))

    def forward(self, x_att: Tensor, vecQuestions: Tensor) -> Tensor:
        """forward computation of LCGNClassiferHead.

        def forward(self, x_att, vecQuestions):
            eQ = self.outQuestion(vecQuestions)
            if self.OUT_QUESTION_MUL:
                features = torch.cat([x_att, eQ, x_att * eQ], dim=-1)
            else:
                features = torch.cat([x_att, eQ], dim=-1)
            logits = self.classifier_layer(features)
            return logits
            Args:
              x_att: shape of batch_size x 512
              vecQuestions: question feature shaped of batch_size x 512

            Returns:
              Tensor: the predicted values
        """
        eQ = self.outQuestion(vecQuestions)
        if self.OUT_QUESTION_MUL:
            features = torch.cat([x_att, eQ, x_att * eQ], dim=-1)
        else:
            features = torch.cat([x_att, eQ], dim=-1)
        logits = self.classifier_layer(features)
        return logits


@HEADS.register_module()
class TripleLinearHead(ClassifierHead):
    """The three-branch classifier in https://arxiv.org/abs/2004.11883:

    During training, all three branches will produce the prediction on its own. During inference, only the fused branch
    is used to predict the answers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linears = nn.ModuleList([nn.Linear(self.in_dim, self.out_dim) for _ in range(3)])

    def forward(self, joint_embedding: torch.Tensor) -> torch.Tensor:
        if self.training:
            feat = [self.linears[i](joint_embedding[:, i]) for i in range(3)]
            return torch.stack(feat, dim=1)
        else:
            return self.linears[0](joint_embedding)


@HEADS.register_module()
class LinearHead(ClassifierHead):
    """The three-branch classifier in https://arxiv.org/abs/2004.11883:

    During training, all three branches will produce the prediction on its own. During inference, only the fused branch
    is used to predict the answers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


@HEADS.register_module()
class WeightNormClassifierHead(ClassifierHead):

    def __init__(self, hidden_dim, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [
            weight_norm(nn.Linear(self.in_dim, hidden_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hidden_dim, self.out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


@HEADS.register_module()
class R2CHead(ClassifierHead):

    def __init__(self, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [
            nn.Dropout(dropout, inplace=True),
            nn.Linear(self.in_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(self.out_dim, 1),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
