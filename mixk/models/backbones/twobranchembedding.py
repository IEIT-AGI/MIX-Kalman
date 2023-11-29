from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

from mixk.models.embedding.textembedding import AttnPool1d, MovieMcanMultiHeadAttention
from ..builder import BACKBONES


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


@BACKBONES.register_module()
class TwoBranchEmbedding(nn.Module):
    """Attach MoVie into MCAN model as a counting module in
    https://arxiv.org/abs/2004.11883."""

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        hidden_dim = kwargs.get('hidden_dim', 512)
        self.sga = SGAEmbedding(embedding_dim, **kwargs)
        self.sga_pool = AttnPool1d(hidden_dim, 1)
        self.cbn = CBNEmbedding(embedding_dim, **kwargs)
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sga = self.sga(x, y, x_mask, y_mask)
        x_sga = self.sga_pool(x_sga, x_sga, x_mask).squeeze(1)
        x_cbn = self.cbn(x, v)

        return x_sga, x_cbn


class CBNEmbedding(nn.Module):
    """MoVie bottleneck layers from https://arxiv.org/abs/2004.11883."""

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        cond_dim = kwargs['cond_dim']
        num_layers = kwargs['cbn_num_layers']
        compressed = kwargs.get('compressed', True)
        use_se = kwargs.get('use_se', True)

        self.out_dim = 1024
        self.layer_norm = nn.LayerNorm(self.out_dim)
        cbns = []
        for i in range(num_layers):
            if embedding_dim != self.out_dim:
                downsample = nn.Conv2d(embedding_dim, self.out_dim, kernel_size=1, stride=1, bias=False)
                cbns.append(
                    MovieBottleneck(
                        embedding_dim,
                        self.out_dim // 4,
                        cond_dim,
                        downsample=downsample,
                        compressed=compressed,
                        use_se=use_se,
                    ))
            else:
                cbns.append(
                    MovieBottleneck(
                        embedding_dim,
                        self.out_dim // 4,
                        cond_dim,
                        compressed=compressed,
                        use_se=use_se,
                    ))
            embedding_dim = self.out_dim
        self.cbns = nn.ModuleList(cbns)
        self._init_layers()

    def _init_layers(self) -> None:
        for cbn in self.cbns:
            cbn.init_layers()

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        for cbn in self.cbns:
            x, _ = cbn(x, v)

        x = self.layer_norm(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(3).squeeze(2))

        return x


class SelfGuidedAttention(nn.Module):

    def __init__(self, dim: int, num_attn: int, dropout: float):
        super().__init__()
        self.multi_head_attn = nn.ModuleList(
            [MovieMcanMultiHeadAttention(dim, num_attn, dropout=0.1) for _ in range(2)])
        self.fcn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4 * dim, dim),
        )
        self.drop_mha = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        self.ln_mha = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.drop_fcn = nn.Dropout(p=dropout)
        self.ln_fcn = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.ln_mha[0](x + self.drop_mha[0](self.multi_head_attn[0](x, x, x, x_mask)))
        x = self.ln_mha[1](x + self.drop_mha[1](self.multi_head_attn[1](x, y, y, y_mask)))
        x = self.ln_fcn(x + self.drop_fcn(self.fcn(x)))

        return x


class SGAEmbedding(nn.Module):
    """Decoder block implementation in MCAN
    https://arxiv.org/abs/1906.10770."""

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        num_attn = kwargs['num_attn']
        num_layers = kwargs['num_layers']
        dropout = kwargs.get('dropout', 0.1)
        hidden_dim = kwargs.get('hidden_dim', 512)

        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.self_guided_attns = nn.ModuleList(
            [SelfGuidedAttention(hidden_dim, num_attn, dropout) for _ in range(num_layers)])
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, -1).transpose(1, 2).contiguous()  # b x (h*w) x c

        x = self.linear(x)

        for self_guided_attn in self.self_guided_attns:
            x = self_guided_attn(x, y, x_mask, y_mask)

        return x


class ChannelPool(nn.Module):
    """Average pooling in the channel dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)


class SEModule(nn.Module):
    """Squeeze-and-Excitation module from https://arxiv.org/pdf/1709.01507.pdf.

    Args:
        dim: the original hidden dim.
        sqrate: the squeeze rate in hidden dim.
    Returns:
        New features map that channels are gated
        by sigmoid weights from SE module.
    """

    def __init__(self, dim: int, sqrate: float):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim // sqrate, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // sqrate, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.attn = nn.Sequential(
            ChannelPool(),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.se(x)

        return x * self.attn(x)


class Modulation(nn.Module):

    def __init__(self, num_features: int, num_cond_features: int, compressed: bool = True):
        super().__init__()
        self.linear = nn.Linear(num_cond_features, num_features)
        self.conv = (
            nn.Conv2d(num_features, 256, kernel_size=1) if compressed else nn.Conv2d(
                num_features, num_features, kernel_size=1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = self.linear(cond).unsqueeze(2).unsqueeze(3)

        return self.conv(x * cond)


class MovieBottleneck(nn.Module):
    """Standard ResNet bottleneck with MoVie modulation in
    https://arxiv.org/abs/2004.11883 The code is inspired from
    https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        cond_planes: int = None,
        stride: int = 1,
        downsample: Optional[Type[nn.Module]] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        stride_in_1x1: bool = False,
        compressed: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        if norm_layer is None:
            self.norm_layer = FrozenBatchNorm2d
        else:
            self.norm_layer = norm_layer
        self.cond_planes = cond_planes
        self.planes = planes
        self.inplanes = inplanes

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(inplanes, self.width, stride_1x1)
        self.bn1 = self.norm_layer(self.width)
        self.conv2 = conv3x3(self.width, self.width, stride_3x3, groups, dilation)
        self.bn2 = self.norm_layer(self.width)
        self.conv3 = conv1x1(self.width, planes * self.expansion)
        self.bn3 = self.norm_layer(self.planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = None

        self.compressed = compressed
        self.use_se = use_se

    def init_layers(self):
        if self.cond_planes:
            self.cond = Modulation(self.inplanes, self.cond_planes, compressed=self.compressed)
            self.se = SEModule(self.planes * self.expansion, 4) if self.use_se else None

    def forward(self,
                x: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        identity = x

        if self.cond_planes and self.compressed:
            x = self.conv1(x) + self.cond(x, cond)
        elif self.cond_planes and not self.compressed:
            x += self.cond(x, cond)
            x = self.conv1(x)
        else:
            x = self.conv1(x)

        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            shortcut = self.downsample(identity)
        else:
            shortcut = identity

        if self.se:
            out = self.se(out)

        out += shortcut
        out = self.relu(out)

        return out, cond
