import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from ..builder import BACKBONES
from ..combine_layers import ModalCombineLayer


class AttentionLayer(nn.Module):

    def __init__(self, image_dim, question_dim, **kwargs):
        super().__init__()

        combine_type = kwargs['modal_combine']['type']
        combine_params = kwargs['modal_combine']['params']
        modal_combine_layer = ModalCombineLayer(combine_type, image_dim, question_dim, **combine_params)

        transform_type = kwargs['transform']['type']
        transform_params = kwargs['transform']['params']
        transform_layer = TransformLayer(transform_type, modal_combine_layer.out_dim, **transform_params)

        normalization = kwargs['normalization']

        self.module = TopDownAttention(modal_combine_layer, transform_layer, normalization)

        if hasattr(self.module, 'out_dim'):
            self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@BACKBONES.register_module()
class ImageFeatureEmbedding(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    def __init__(self, img_dim, question_dim, **kwargs):
        super().__init__()

        self.image_attention_model = AttentionLayer(img_dim, question_dim, **kwargs)
        self.out_dim = self.image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, order_vectors=None):
        # if extra is None:
        #     extra = {}
        # N x K x n_att
        attention = self.image_attention_model(image_feat_variable, question_embedding, image_dims)
        att_reshape = attention.permute(0, 2, 1)

        # order_vectors = getattr(extra, "order_vectors", None)

        if order_vectors is not None:
            image_feat_variable = torch.cat([image_feat_variable, order_vectors], dim=-1)
        tmp_embedding = torch.bmm(att_reshape, image_feat_variable)  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding, attention


class TransformLayer(nn.Module):

    def __init__(self, transform_type, in_dim, out_dim, hidden_dim=None):
        super().__init__()

        if transform_type == 'linear':
            self.module = LinearTransform(in_dim, out_dim)
        elif transform_type == 'conv':
            self.module = ConvTransform(in_dim, out_dim, hidden_dim)
        else:
            raise NotImplementedError('Unknown post combine transform type: %s' % transform_type)
        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TopDownAttention(nn.Module):
    EPS = 1.0e-08

    def __init__(self, combination_layer, transform_module, normalization):
        super().__init__()
        self.combination_layer = combination_layer
        self.normalization = normalization
        self.transform = transform_module
        self.out_dim = self.transform.out_dim

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.size()
        tmp1 = attention.new_zeros(num_loc)
        tmp1[:num_loc] = torch.arange(0, num_loc, dtype=attention.dtype).unsqueeze(dim=0)

        tmp1 = tmp1.expand(batch_size, num_loc)
        tmp2 = image_locs.type(tmp1.type())
        tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = mask.unsqueeze(dim=2).expand_as(attention)
        attention = attention.masked_fill(mask, 0)
        return attention

    def forward(self, image_feat, question_embedding, image_locs=None):
        # N x K x joint_dim
        joint_feature = self.combination_layer(image_feat, question_embedding)
        # N x K x n_att
        raw_attn = self.transform(joint_feature)

        if self.normalization.lower() == 'softmax':
            attention = nn.functional.softmax(raw_attn, dim=1)
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
                masked_attention_sum = torch.sum(masked_attention, dim=1, keepdim=True)
                masked_attention_sum += masked_attention_sum.eq(0).float() + self.EPS
                masked_attention = masked_attention / masked_attention_sum
            else:
                masked_attention = attention

        elif self.normalization.lower() == 'sigmoid':
            attention = torch.sigmoid(raw_attn)
            masked_attention = attention
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)

        return masked_attention


class LinearTransform(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lc = weight_norm(nn.Linear(in_features=in_dim, out_features=out_dim), dim=None)
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)


class ConvTransform(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1)
        self.out_dim = out_dim

    def forward(self, x):
        if len(x.size()) == 3:  # N x k xdim
            # N x dim x k x 1
            x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
        elif len(x.size()) == 2:  # N x dim
            # N x dim x 1 x 1
            x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)

        iatt_conv1 = self.conv1(x_reshape)  # N x hidden_dim x * x 1
        iatt_relu = nn.functional.relu(iatt_conv1)
        iatt_conv2 = self.conv2(iatt_relu)  # N x out_dim x * x 1

        if len(x.size()) == 3:
            iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
        elif len(x.size()) == 2:
            iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)

        return iatt_conv3
