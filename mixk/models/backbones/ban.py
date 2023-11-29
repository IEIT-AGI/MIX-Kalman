import torch.nn as nn

from ..builder import BACKBONES
from ..combine_layers import BCNet, BiAttention, FCNet


@BACKBONES.register_module()
class BAN_BACKBONE(nn.Module):

    def __init__(self, v_dim, num_hidden, gamma, k, activation, dropout):
        super().__init__()
        v_att = BiAttention(v_dim, num_hidden, num_hidden, gamma)
        self.gamma = gamma
        b_net = []
        q_prj = []

        for _ in range(gamma):
            b_net.append(BCNet(v_dim, num_hidden, num_hidden, None, k=k))

            q_prj.append(FCNet(
                dims=[num_hidden, num_hidden],
                act=activation,
                dropout=dropout,
            ))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.v_att = v_att

    def forward(self, v, q_emb):
        b_emb = [0] * self.gamma
        att, logits = self.v_att.forward_all(v, q_emb)
        for g in range(self.gamma):
            g_att = att[:, g, :, :]
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, g_att)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        return q_emb
