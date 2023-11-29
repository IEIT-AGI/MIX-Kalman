from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform
from torch.autograd import Variable
from ..builder import BACKBONES


def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    data = [
        xv_min.unsqueeze(0),
        yv_min.unsqueeze(0),
        xv_max.unsqueeze(0),
        yv_max.unsqueeze(0),
        xv_ctr.unsqueeze(0),
        yv_ctr.unsqueeze(0),
        hmap.unsqueeze(0),
        wmap.unsqueeze(0)
    ]
    coord = torch.autograd.Variable(torch.cat(data, dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord


@BACKBONES.register_module()
class CMRIN_BACKBONE(nn.Module):

    def __init__(self,
                 emb_size=512,
                 jemb_drop_out=0.1,
                 NFilm=2,
                 light=False,
                 coordmap=True,
                 intmd=False,
                 mstage=False,
                 convlstm=False,
                 tunebert=True,
                 leaky=False):
        super().__init__()

        self.emb_size = emb_size
        self.NFilm = NFilm
        self.intmd = intmd
        self.mstage = mstage
        self.convlstm = convlstm
        self.tunebert = tunebert
        self.textdim = 768  # 'bert-base-uncased'

        # Mapping module
        self.mapping_lang = torch.nn.Sequential(
            nn.Linear(self.textdim, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(),
            nn.Dropout(jemb_drop_out),
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(),
        )

        self.light = light
        embin_size = emb_size * 2
        self.coordmap = coordmap
        if self.coordmap:
            embin_size += 8

        self.mapping_visu = nn.Sequential(
            OrderedDict([('0', ConvBatchNormReLU(1024, emb_size, 1, 1, 0, 1, leaky=False)),
                         ('1', ConvBatchNormReLU(512, emb_size, 1, 1, 0, 1, leaky=False)),
                         ('2', ConvBatchNormReLU(256, emb_size, 1, 1, 0, 1, leaky=False))]))
        if self.light:
            self.fcn_emb = nn.Sequential(
                OrderedDict([
                    ('0', torch.nn.Sequential(ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky), )),
                    ('1', torch.nn.Sequential(ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky), )),
                    ('2', torch.nn.Sequential(ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky), )),
                ]))
            self.fcn_out = nn.Sequential(
                OrderedDict([
                    ('0', torch.nn.Sequential(nn.Conv2d(emb_size, 3 * 5, kernel_size=1), )),
                    ('1', torch.nn.Sequential(nn.Conv2d(emb_size, 3 * 5, kernel_size=1), )),
                    ('2', torch.nn.Sequential(nn.Conv2d(emb_size, 3 * 5, kernel_size=1), )),
                ]))
        else:
            self.fcn_emb = nn.Sequential(
                OrderedDict([
                    ('0',
                     torch.nn.Sequential(
                         ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                         ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                         ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                     )),
                    ('1',
                     torch.nn.Sequential(
                         ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                         ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                         ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                     )),
                    ('2',
                     torch.nn.Sequential(
                         ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                         ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                         ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                     )),
                ]))
            self.fcn_out = nn.Sequential(
                OrderedDict([
                    ('0',
                     torch.nn.Sequential(
                         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
                         nn.Conv2d(emb_size // 2, 3 * 5, kernel_size=1),
                     )),
                    ('1',
                     torch.nn.Sequential(
                         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
                         nn.Conv2d(emb_size // 2, 3 * 5, kernel_size=1),
                     )),
                    ('2',
                     torch.nn.Sequential(
                         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
                         nn.Conv2d(emb_size // 2, 3 * 5, kernel_size=1),
                     )),
                ]))

    def forward(self, raw_flang, raw_fvisu):
        fvisu = []
        for ii in range(len(raw_fvisu)):
            fvisu.append(self.mapping_visu._modules[str(ii)](raw_fvisu[ii]))
            fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)
        flang = self.mapping_lang(raw_flang)
        flang = F.normalize(flang, p=2, dim=1)
        batch_size = raw_flang.size(0)
        flangvisu = []
        for ii in range(len(fvisu)):
            flang_tile = flang.view(flang.size(0), flang.size(1), 1, 1). \
                repeat(1, 1, fvisu[ii].size(2), fvisu[ii].size(3))
            if self.coordmap:
                coord = generate_coord(batch_size, fvisu[ii].size(2), fvisu[ii].size(3))
                flangvisu.append(torch.cat([fvisu[ii], flang_tile, coord], dim=1))
            else:
                flangvisu.append(torch.cat([fvisu[ii], flang_tile], dim=1))

        # fcn
        intmd_fea, outbox = [], []
        for ii in range(len(fvisu)):
            intmd_fea.append(self.fcn_emb._modules[str(ii)](flangvisu[ii]))
            outbox.append(self.fcn_out._modules[str(ii)](intmd_fea[ii]))
        return outbox


class ConvBatchNormReLU(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        leaky=False,
        relu=True,
        instance=False,
    ):
        super(ConvBatchNormReLU, self).__init__()
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        if instance:
            self.add_module(
                'bn',
                nn.InstanceNorm2d(num_features=out_channels),
            )
        else:
            self.add_module(
                'bn',
                nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.999, affine=True),
            )

        if leaky:
            self.add_module('relu', nn.LeakyReLU(0.1))
        elif relu:
            self.add_module('relu', nn.ReLU())

    def forward(self, x):
        return super(ConvBatchNormReLU, self).forward(x)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self,
                 input_size,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 num_layers,
                 batch_first=False,
                 bias=True,
                 return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None.

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


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


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            init_params(m.weight)
