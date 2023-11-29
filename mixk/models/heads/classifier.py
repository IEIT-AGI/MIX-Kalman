import torch
import torch.nn as nn

from mixk.models.backbones.lcgn_backbone import Linear
from mixk.models.combine_layers import ReLUWithWeightNormFC
from ..builder import HEADS
from torch.nn.utils.weight_norm import weight_norm


@HEADS.register_module()
class ClassifierLayer(nn.Module):

    def __init__(self, classifier_type, in_dim, out_dim, **kwargs):
        super().__init__()

        if classifier_type == 'weight_norm':
            self.module = WeightNormClassifier(in_dim, out_dim, **kwargs)
        elif classifier_type == 'logit':
            self.module = LogitClassifier(in_dim, out_dim, **kwargs)
        elif classifier_type == 'language_decoder':
            # self.module = LanguageDecoder(in_dim, out_dim, **kwargs)
            self.module = None
        elif classifier_type == 'bert':
            self.module = BertClassifierHead(in_dim, out_dim, kwargs.get('config', None)).module
        elif classifier_type == 'mlp':
            self.module = MLPClassifer(in_dim, out_dim, **kwargs)
        elif classifier_type == 'linear':
            self.module = nn.Linear(in_dim, out_dim)

        elif classifier_type == 'triple_linear':
            self.module = TripleLinear(in_dim, out_dim)

        elif classifier_type == 'lcgn':
            self.module = LCGNClassifer(in_dim, out_dim, **kwargs)
        else:
            raise NotImplementedError('Unknown classifier type: %s' % classifier_type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BertClassifierHead(nn.Module):

    def __init__(self, in_dim=768, out_dim=2, config=None, *args, **kwargs):
        super().__init__()
        from transformers.modeling_bert import BertPredictionHeadTransform

        if config is None:
            from transformers.configuration_bert import BertConfig

            config = BertConfig.from_pretrained('bert-base-uncased')

        assert config.hidden_size == in_dim

        self.module = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            BertPredictionHeadTransform(config),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MLPClassifer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=None,
        num_layers=0,
        dropout=0.5,
        hidden_act='relu',
        batch_norm=True,
        **kwargs,
    ):
        super().__init__()
        from mixk.utils.modeling import ACT2FN

        activation = ACT2FN[hidden_act]
        self.layers = nn.ModuleList()

        if hidden_dim is None:
            hidden_dim = in_dim

        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(activation())
            self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LogitClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        input_dim = in_dim
        num_ans_candidates = out_dim
        text_non_linear_dim = kwargs['text_hidden_dim']
        image_non_linear_dim = kwargs['img_hidden_dim']

        self.f_o_text = ReLUWithWeightNormFC(input_dim, text_non_linear_dim)
        self.f_o_image = ReLUWithWeightNormFC(input_dim, image_non_linear_dim)
        self.linear_text = nn.Linear(text_non_linear_dim, num_ans_candidates)
        self.linear_image = nn.Linear(image_non_linear_dim, num_ans_candidates)

        if 'pretrained_image' in kwargs and kwargs['pretrained_text'] is not None:
            self.linear_text.weight.data.copy_(torch.from_numpy(kwargs['pretrained_text']))

        if 'pretrained_image' in kwargs and kwargs['pretrained_image'] is not None:
            self.linear_image.weight.data.copy_(torch.from_numpy(kwargs['pretrained_image']))

    def forward(self, joint_embedding):
        text_val = self.linear_text(self.f_o_text(joint_embedding))
        image_val = self.linear_image(self.f_o_image(joint_embedding))
        logit_value = text_val + image_val

        return logit_value


class WeightNormClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super().__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hidden_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hidden_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class LCGNClassifer(nn.Module):

    def __init__(self, in_dim, out_dim, OUT_QUESTION_MUL, CMD_DIM, outputDropout):
        super().__init__()

        self.OUT_QUESTION_MUL = OUT_QUESTION_MUL
        self.outQuestion = Linear(CMD_DIM, CMD_DIM)
        in_dim = 3 * in_dim if OUT_QUESTION_MUL else 2 * in_dim
        self.classifier_layer = nn.Sequential(
            nn.Dropout(1 - outputDropout), Linear(in_dim, CMD_DIM), nn.ELU(), nn.Dropout(1 - outputDropout),
            Linear(CMD_DIM, out_dim))

    def forward(self, x_att, vecQuestions):
        eQ = self.outQuestion(vecQuestions)
        if self.OUT_QUESTION_MUL:
            features = torch.cat([x_att, eQ, x_att * eQ], dim=-1)
        else:
            features = torch.cat([x_att, eQ], dim=-1)
        logits = self.classifier_layer(features)
        return logits


class TripleLinear(nn.Module):
    """The three-branch classifier in https://arxiv.org/abs/2004.11883:

    During training, all three branches will produce the prediction on its own. During inference, only the fused branch
    is used to predict the answers.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(3)])

    def forward(self, joint_embedding: torch.Tensor) -> torch.Tensor:
        if self.training:
            feat = [self.linears[i](joint_embedding[:, i]) for i in range(3)]
            return torch.stack(feat, dim=1)

        return self.linears[0](joint_embedding)
