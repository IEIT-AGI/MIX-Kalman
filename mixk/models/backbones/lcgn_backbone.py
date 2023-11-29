import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from ..builder import BACKBONES


class Linear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # compatible with xavier_initializer in TensorFlow
        fan_avg = (self.in_features + self.out_features) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


activations = {
    'NON': lambda x: x,
    'TANH': torch.tanh,
    'SIGMOID': F.sigmoid,
    'RELU': F.relu,
    'ELU': F.elu,
}


def apply_mask1d(attention: Tensor, image_locs: Tensor) -> Tensor:
    """apply mask on image locations.

    Args:
        attention: the attention results to generate textual command
        image_locs: question length of samples

    Returns:
        Tensor: mask for 1 dimension
    """
    attention = attention.float()
    batch_size, num_loc = attention.size()  # (127, 128)
    tmp1 = attention.new_zeros(num_loc)  # 128
    tmp1[:num_loc] = torch.arange(0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask = torch.ge(tmp1, tmp2)
    attention = attention.masked_fill(mask, -1e30)
    return attention


def apply_mask2d(attention: Tensor, image_locs: Tensor) -> Tensor:
    """mask the unshowed entity.

    Args:
        attention:edge score between different entities
        image_locs:the entity number of samples

    Returns:
        Tensor: mask for two dimension
    """
    attention = attention.float()
    batch_size, num_loc, _ = attention.size()
    tmp1 = attention.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask1d = torch.ge(tmp1, tmp2)
    mask2d = mask1d[:, None, :] | mask1d[:, :, None]
    attention = attention.masked_fill(mask2d, -1e30)
    return attention


def generate_scaled_var_drop_mask(shape: Tensor, keep_prob: Tensor) -> Tensor:
    """generate a mask tensor respect to the context feature.

    Args:
        shape: : context feature's shape, shaped of batch_size x 36 x 512
        keep_prob: probability to keep when the generated random number is larger or equal to keep_prob

    Returns:
        Tensor: generated mask.
    """
    assert keep_prob > 0. and keep_prob <= 1.
    mask = torch.rand(shape, device='cpu').le(keep_prob).cuda()  # cuda
    mask = mask.float() / keep_prob
    return mask


@BACKBONES.register_module()
class LCGN_BACKBONE(nn.Module):

    def __init__(self, stem_linear: bool, D_FEAT: int, CTX_DIM: int, CMD_DIM: int, MSG_ITER_NUM: int,
                 stemDropout: float, readDropout: float, memoryDropout: float, CMD_INPUT_ACT: str,
                 STEM_NORMALIZE: bool) -> None:
        """the initialization of LCGN backbone.

        Args:
            stem_linear: bool to identify if do linear operation in loc_ctx_init
            D_FEAT: the origin dimension of the local feature
            CTX_DIM: the dimension of the context feature
            CMD_DIM: the dimension of the command vector
            MSG_ITER_NUM: the number of message passing iteration
            stemDropout: dropout rate in stem
            readDropout: dropout rate in the first input layer
            memoryDropout: drop out during the updation of context feature
            CMD_INPUT_ACT: the activation after the embedding of command vector
            STEM_NORMALIZE: bool to identify if do the normalization of image features in loc_ctx_init
        """
        super().__init__()

        self.STEM_LINEAR = stem_linear
        self.D_FEAT = D_FEAT
        self.CTX_DIM = CTX_DIM
        self.CMD_DIM = CMD_DIM
        self.stemDropout = stemDropout
        self.readDropout = readDropout
        self.memoryDropout = memoryDropout

        self.MSG_ITER_NUM = MSG_ITER_NUM
        self.CMD_INPUT_ACT = CMD_INPUT_ACT
        self.STEM_NORMALIZE = STEM_NORMALIZE

        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()

    def build_loc_ctx_init(self):
        assert self.STEM_LINEAR
        if self.STEM_LINEAR:
            self.initKB = Linear(self.D_FEAT, self.CTX_DIM)
            self.x_loc_drop = nn.Dropout(1 - self.stemDropout)

        self.initMem = nn.Parameter(torch.randn(1, 1, self.CTX_DIM))

    def build_extract_textual_command(self):
        self.qInput = Linear(self.CMD_DIM, self.CMD_DIM)
        for t in range(self.MSG_ITER_NUM):
            qInput_layer2 = Linear(self.CMD_DIM, self.CMD_DIM)
            setattr(self, 'qInput%d' % t, qInput_layer2)
        self.cmd_inter2logits = Linear(self.CMD_DIM, 1)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - self.readDropout)
        self.project_x_loc = Linear(self.CTX_DIM, self.CTX_DIM)
        self.project_x_ctx = Linear(self.CTX_DIM, self.CTX_DIM)
        self.queries = Linear(3 * self.CTX_DIM, self.CTX_DIM)
        self.keys = Linear(3 * self.CTX_DIM, self.CTX_DIM)
        self.vals = Linear(3 * self.CTX_DIM, self.CTX_DIM)
        self.proj_keys = Linear(self.CMD_DIM, self.CTX_DIM)
        self.proj_vals = Linear(self.CMD_DIM, self.CTX_DIM)
        self.mem_update = Linear(2 * self.CTX_DIM, self.CTX_DIM)
        self.combine_kb = Linear(2 * self.CTX_DIM, self.CTX_DIM)

    def forward(self, images: Tensor, q_encoding: Tensor, lstm_outputs: Tensor, q_length: Tensor,
                entity_num: Tensor) -> Tensor:
        """The backbone network including message passing process based on the
        graph constructed by detected areas, which are seen as nodes in a
        graph.

        Args:
            images:  image features from object detection models
            q_encoding: the question encoding of final hidden states?
            lstm_outputs: the output of bilstm
            q_length: question length, which is used for mask of message passing
            entity_num: the number of entity comes from object detection model

        Returns:
            Tensor: the output feature containing local representation and context representation
        """
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)
        for t in range(self.MSG_ITER_NUM):
            x_ctx = self.run_message_passing_iter(q_encoding, lstm_outputs, q_length, x_loc, x_ctx, x_ctx_var_drop,
                                                  entity_num, t)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        return x_out

    def extract_textual_command(self, q_encoding: Tensor, lstm_outputs: Tensor, q_length: Tensor, t: int) -> Tensor:
        """to extract command feature from text.

        Args:
            q_encoding: the question encoding of final hidden states, sized of batch_size x 512
            lstm_outputs: the output of bilstm, shaped of batch_size x 128 x 512
            q_length: question length, which is used for mask of message passing, (batch_size,)
            t: the iterable number to record the message passing number

        Returns:
            Tensor: textual command feature during the t iteration.
        """
        qInput_layer2 = getattr(self, 'qInput%d' % t)
        act_fun = activations[self.CMD_INPUT_ACT]
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = apply_mask1d(raw_att, q_length)  # (batch_size, 128)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)  # (127, 1, 128) (127,128,512)
        return cmd

    def propagate_message(self, cmd: Tensor, x_loc: Tensor, x_ctx: Tensor, x_ctx_var_drop: Tensor,
                          entity_num: Tensor) -> Tensor:
        """let's do message passing for one time.

        Args:
            cmd: textual command feature during the t iteration. (batch_size, 512)
            x_loc: local feature is initialized from image feature
            x_ctx: context feature, is initialized randomly, which is same in all positions
            x_ctx_var_drop: the mask of context feature
            entity_num: the number of entity comes from object detection model

        Returns:
            Tensor: the new context feature after one time message propagation
        """
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat([x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)  # (127, 36, 1536)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        edge_score = (torch.bmm(queries, torch.transpose(keys, 1, 2)) / np.sqrt(self.CTX_DIM))
        edge_score = apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)
        queries = self.queries(x_joint)  # (127, 36, 512)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]  # (batch_size, 36, 512)
        edge_score = (torch.bmm(queries, torch.transpose(keys, 1, 2)) / np.sqrt(self.CTX_DIM))
        edge_score = apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)  # (batch_size, 36, 36)
        message = torch.bmm(edge_prob, vals)  # (batch_size, 36, 512)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(self, q_encoding: Tensor, lstm_outputs: Tensor, q_length: Tensor, x_loc: Tensor,
                                 x_ctx: Tensor, x_ctx_var_drop: Tensor, entity_num: Tensor, t: int) -> Tensor:
        """one time for message passing, let's go for one time.

        Args:
            q_encoding: the question encoding of final hidden states?
            lstm_outputs: the output of bilstm
            q_length: question length, which is used for mask of message passing
            x_loc: location information is initialized from image feature
            x_ctx: the context feature is initialized randomly, which is same in all positions
            x_ctx_var_drop: the mask of context feature
            entity_num: the number of entity comes from object detection model
            t: the iterable number to record the message passing number
        Returns:
            Tensor: the new context feature after one time message propagation
        """
        cmd = self.extract_textual_command(q_encoding, lstm_outputs, q_length, t)
        x_ctx = self.propagate_message(cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx

    def loc_ctx_init(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """

        Args:
            images: all the features of objects, batch_size x36 x 2112

        Returns:
            Tuple[Tensor, Tensor, Tensor]: a tuple of three Tensors;
            x_loc: location information is initialized from image feature;
            x_ctx: the context feature is initialized randomly, which is same in all positions;
            x_ctx_var_drop: the mask of context feature;

        """
        if self.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if self.STEM_LINEAR:
            # print(self.initKB.state_dict()['weight'].size()[-1])
            # print(images.size()[-1])
            x_loc = self.initKB(images)

            x_loc = self.x_loc_drop(x_loc)

        # if self.STEM_RENORMALIZE:
        #     x_loc = F.normalize(x_loc, dim=-1)

        x_ctx = self.initMem.expand(x_loc.size())
        x_ctx_var_drop = generate_scaled_var_drop_mask(
            x_ctx.size(), keep_prob=(self.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop
