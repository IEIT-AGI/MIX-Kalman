from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal, kaiming_uniform

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
class ReSC_BACKBONE(nn.Module):

    def __init__(self,
                 emb_size=512,
                 jemb_drop_out=0.1,
                 NFilm=2,
                 fusion='prod',
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

        # #### test forward
        # seed = 13
        # import random
        # import numpy as np
        # random.seed(seed)
        # np.random.seed(seed + 1)
        # torch.manual_seed(seed + 2)
        # torch.cuda.manual_seed_all(seed + 3)
        # jemb_drop_out = 0.0
        # ###########

        # Mapping module
        self.mapping_visu = ConvBatchNormReLU(512 if self.convlstm else 256, emb_size, 1, 1, 0, 1, leaky=leaky)

        self.mapping_lang = torch.nn.Sequential(
            nn.Linear(self.textdim, emb_size), nn.ReLU(), nn.Dropout(jemb_drop_out), nn.Linear(emb_size, emb_size),
            nn.ReLU())

        textdim = emb_size
        self.film = FiLMedConvBlock_multihop(
            NFilm=NFilm,
            textdim=textdim,
            visudim=emb_size,
            emb_size=emb_size,
            fusion=fusion,
            intmd=(intmd or mstage or convlstm))

        # output head
        output_emb = emb_size
        if self.mstage:
            # selfn_out = nn.ModuleDict()
            modules = OrderedDict()
            for n in range(0, NFilm):
                modules['out%d' % n] = torch.nn.Sequential(
                    ConvBatchNormReLU(output_emb, output_emb // 2, 1, 1, 0, 1, leaky=leaky),
                    nn.Conv2d(output_emb // 2, 9 * 5, kernel_size=1))
            self.fcn_out.update(modules)
        else:
            if self.intmd:
                output_emb = emb_size * NFilm
            if self.convlstm:
                output_emb = emb_size
                self.global_out = ConvLSTM(
                    input_size=(32, 32),
                    input_dim=emb_size,
                    hidden_dim=[emb_size],
                    kernel_size=(1, 1),
                    num_layers=1,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)
            self.fcn_out = torch.nn.Sequential(
                ConvBatchNormReLU(output_emb, output_emb // 2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(output_emb // 2, 9 * 5, kernel_size=1))

    def forward(self, raw_fword, raw_fvisu, word_mask):
        fvisu = self.mapping_visu(raw_fvisu)
        raw_fvisu = F.normalize(fvisu, p=2, dim=1)
        fword = Variable(torch.zeros(raw_fword.shape[0], raw_fword.shape[1], self.emb_size).cuda())
        batch_size = raw_fword.size(0)

        for ii in range(raw_fword.shape[0]):
            ntoken = (word_mask[ii] != 0).sum()
            fword[ii, :ntoken, :] = F.normalize(self.mapping_lang(raw_fword[ii, :ntoken, :]), p=2, dim=1)
            # [CLS], [SEP]
            # fword[ii,1:ntoken-1,:] = F.normalize(
            # self.mapping_lang(raw_fword[ii,1:ntoken-1,:].view(-1,self.textdim)), p=2, dim=1)
        raw_fword = fword

        coord = generate_coord(batch_size, raw_fvisu.size(2), raw_fvisu.size(3))
        x, attnscore_list = self.film(raw_fvisu, raw_fword, coord, fsent=None, word_mask=word_mask)
        if self.mstage:
            outbox = []
            for film_ii in range(len(x)):
                outbox.append(self.fcn_out['out%d' % film_ii](x[film_ii]))
        elif self.convlstm:
            x = torch.stack(x, dim=1)
            output, state = self.global_out(x)
            # output, hidden = output[-1], state[-1][0]
            # output, hidden, cell = output[-1], state[-1][0], state[-1][1]
            output, hidden = output[-1], state[-1][0]
            outbox = [self.fcn_out(hidden)]
        else:
            x = torch.stack(x, dim=1).view(batch_size, -1, raw_fvisu.size(2), raw_fvisu.size(3))
            outbox = [self.fcn_out(x)]
        return outbox, attnscore_list  # list


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


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def forward(self, x, gammas, betas):
        # gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        # betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FiLMedConvBlock_context(nn.Module):

    def __init__(self,
                 with_residual=True,
                 with_batchnorm=True,
                 with_cond=[False],
                 dropout=0,
                 num_extra_channels=0,
                 extra_channel_freq=1,
                 with_input_proj=1,
                 num_cond_maps=8,
                 kernel_size=1,
                 batchnorm_affine=False,
                 num_layers=1,
                 condition_method='bn-film',
                 debug_every=float('inf'),
                 textdim=768,
                 visudim=512,
                 contextdim=512,
                 emb_size=512,
                 fusion='prod',
                 cont_map=False,
                 lstm=False,
                 baseline=False):
        super(FiLMedConvBlock_context, self).__init__()

        self.cont_map = cont_map  # mapping context with language feature
        self.lstm = lstm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.fusion = fusion
        self.baseline = baseline
        self.film = FiLM()

        if self.cont_map:
            self.sent_map = nn.Linear(768, emb_size)
            self.context_map = nn.Linear(emb_size, emb_size)
        if self.fusion == 'cat':
            self.attn_map = nn.Conv1d(textdim + visudim, emb_size // 2, kernel_size=1)
        elif self.fusion == 'prod':
            assert (textdim == visudim)  # if product fusion
            self.attn_map = nn.Conv1d(visudim, emb_size // 2, kernel_size=1)

        self.attn_score = nn.Conv1d(emb_size // 2, 1, kernel_size=1)
        if self.baseline:
            self.fusion_layer = ConvBatchNormReLU(visudim + textdim + 8, emb_size, 1, 1, 0, 1)
        else:
            self.gamme_decode = nn.Linear(textdim, 2 * emb_size)
            self.conv1 = nn.Conv2d(visudim + 8, emb_size, kernel_size=1)
            # self.bn1 = nn.BatchNorm2d(emb_size)
            self.bn1 = nn.InstanceNorm2d(emb_size)
        init_modules(self.modules())

    def forward(self, fvisu, fword, context_score, fcoord, textattn=None, weight=None, fsent=None, word_mask=None):
        fword = fword.permute(0, 2, 1)
        B, Dvisu, H, W = fvisu.size()
        B, Dlang, N = fword.size()
        B, N = context_score.size()
        assert (Dvisu == Dlang)

        if self.cont_map and fsent is not None:
            fsent = F.normalize(F.relu(self.sent_map(fsent)), p=2, dim=1)
            fcont = torch.matmul(context_score.view(B, 1, N), fword.permute(0, 2, 1)).squeeze(1)
            fcontext = F.relu(self.context_map(fsent * fcont)).unsqueeze(2).repeat(1, 1, N)
            # word attention
            tile_visu = torch.mean(fvisu.view(B, Dvisu, -1), dim=2, keepdim=True).repeat(1, 1, N)
            if self.fusion == 'cat':
                context_tile = torch.cat([tile_visu, fword, fcontext], dim=1)
            elif self.fusion == 'prod':
                context_tile = tile_visu * fword * fcontext
        else:
            # word attention
            tile_visu = torch.mean(fvisu.view(B, Dvisu, -1), dim=2, keepdim=True).repeat(1, 1, N)
            if self.fusion == 'cat':
                context_tile = torch.cat([tile_visu, fword * context_score.view(B, 1, N).repeat(
                    1,
                    Dlang,
                    1,
                )], dim=1)
            elif self.fusion == 'prod':
                context_tile = tile_visu * fword * context_score.view(B, 1, N).repeat(
                    1,
                    Dlang,
                    1,
                )

        attn_feat = F.tanh(self.attn_map(context_tile))
        attn_score = self.attn_score(attn_feat).squeeze(1)
        mask_score = mask_softmax(attn_score, word_mask, lstm=self.lstm)
        attn_lang = torch.matmul(mask_score.view(B, 1, N), fword.permute(0, 2, 1))
        attn_lang = attn_lang.view(B, Dlang).squeeze(1)

        if self.baseline:
            fmodu = self.fusion_layer(
                torch.cat(
                    [fvisu,
                     attn_lang.unsqueeze(2).unsqueeze(2).repeat(1, 1, fvisu.shape[-1], fvisu.shape[-1]), fcoord],
                    dim=1))
        else:
            # lang-> gamma, beta
            film_param = self.gamme_decode(attn_lang)
            film_param = film_param.view(B, 2 * self.emb_size, 1, 1).repeat(1, 1, H, W)
            gammas, betas = torch.split(film_param, self.emb_size, dim=1)
            gammas, betas = F.tanh(gammas), F.tanh(betas)

            # modulate visu feature
            fmodu = self.bn1(self.conv1(torch.cat([fvisu, fcoord], dim=1)))
            fmodu = self.film(fmodu, gammas, betas)
            fmodu = F.relu(fmodu)
        if self.with_residual:
            if weight is None:
                fmodu = fvisu + fmodu
            else:
                weight = weight.view(B, 1, 1, 1).repeat(1, Dvisu, H, W)
                fmodu = (1 - weight) * fvisu + weight * fmodu
        return fmodu, attn_lang, attn_score


class FiLMedConvBlock_multihop(nn.Module):

    def __init__(self,
                 NFilm=2,
                 with_residual=True,
                 with_batchnorm=True,
                 with_cond=[False],
                 dropout=0,
                 num_extra_channels=0,
                 extra_channel_freq=1,
                 with_input_proj=1,
                 num_cond_maps=8,
                 kernel_size=1,
                 batchnorm_affine=False,
                 num_layers=1,
                 condition_method='bn-film',
                 debug_every=float('inf'),
                 textdim=768,
                 visudim=512,
                 emb_size=512,
                 fusion='cat',
                 intmd=False,
                 lstm=False,
                 erasing=0.):
        super(FiLMedConvBlock_multihop, self).__init__()

        self.NFilm = NFilm
        self.emb_size = emb_size
        self.with_residual = with_residual
        self.cont_size = emb_size
        self.fusion = fusion
        self.intmd = intmd
        self.lstm = lstm
        self.erasing = erasing
        if self.fusion == 'cat':
            self.cont_size = emb_size * 2

        self.modulesdict = nn.ModuleDict()
        modules = OrderedDict()
        modules['film0'] = FiLMedConvBlock_context(
            textdim=textdim, visudim=emb_size, contextdim=emb_size, emb_size=emb_size, fusion=fusion, lstm=self.lstm)
        for n in range(1, NFilm):
            modules['conv%d' % n] = ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1)
            modules['film%d' % n] = FiLMedConvBlock_context(
                textdim=textdim,
                visudim=emb_size,
                contextdim=self.cont_size,
                emb_size=emb_size,
                fusion=fusion,
                lstm=self.lstm)
        self.modulesdict.update(modules)

    def forward(self, fvisu, fword, fcoord, weight=None, fsent=None, word_mask=None):
        B, Dvisu, H, W = fvisu.size()
        B, N, Dlang = fword.size()
        intmd_feat, attnscore_list = [], []

        x, _, attn_score = self.modulesdict['film0'](
            fvisu, fword, Variable(torch.ones(B, N).cuda()), fcoord, fsent=fsent, word_mask=word_mask)
        attnscore_list.append(attn_score.view(B, N, 1, 1))
        if self.intmd:
            intmd_feat.append(x)
        if self.NFilm == 1:
            intmd_feat = [x]
        for n in range(1, self.NFilm):
            score_list = [
                mask_softmax(score.squeeze(2).squeeze(2), word_mask, lstm=self.lstm) for score in attnscore_list
            ]

            score = torch.clamp(torch.max(torch.stack(score_list, dim=1), dim=1, keepdim=False)[0], min=0., max=1.)
            x = self.modulesdict['conv%d' % n](x)
            x, _, attn_score = self.modulesdict['film%d' % n](
                x, fword, (1 - score), fcoord, fsent=fsent, word_mask=word_mask)
            attnscore_list.append(attn_score.view(B, N, 1, 1))  # format match div loss in main func
            if self.intmd:
                intmd_feat.append(x)
            elif n == self.NFilm - 1:
                intmd_feat = [x]
        return intmd_feat, attnscore_list
