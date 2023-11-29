import torch
import torch.nn as nn

from ..builder import COMBINE_LAYERS


@COMBINE_LAYERS.register_module()
class BranchCombineLayer(nn.Module):
    """Three-branch fusion module used for fusing MoVie and MCAN in
    https://arxiv.org/abs/2004.11883."""

    def __init__(self, img_dim: int, ques_dim: int):
        super().__init__()
        self.out_dim = img_dim * 2
        self.linear_cga = nn.ModuleList([nn.Linear(img_dim, self.out_dim) for _ in range(2)])
        self.linear_cbn = nn.ModuleList([nn.Linear(img_dim, self.out_dim) for _ in range(2)])
        self.linear_ques = nn.ModuleList([nn.Linear(ques_dim, self.out_dim) for _ in range(2)])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(self.out_dim) for _ in range(3)])

    def forward(self, v_cga: torch.Tensor, v_cbn: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        feat = [
            self.layer_norm[0](self.linear_ques[0](q) + self.linear_cbn[0](v_cbn) + self.linear_cga[0](v_cga)),
            self.layer_norm[1](self.linear_cbn[1](v_cbn)),
            self.layer_norm[2](self.linear_ques[1](q) + self.linear_cga[1](v_cga)),
        ]

        if self.training:
            return torch.stack(feat, dim=1)

        return feat[0]
