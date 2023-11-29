import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import masked_softmax, replace_masked_values

from ..builder import BACKBONES
from .r2c_backbone import R2C_BACKBONE


@BACKBONES.register_module()
class HGL_BACKBONE(R2C_BACKBONE):
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

    def __init__(self, graph_dim=512, **kwargs):
        self.graph_dim = graph_dim
        super(HGL_BACKBONE, self).__init__(**kwargs)
        self.Graph_reasoning = Graph_reasoning(self.graph_dim)

    def forward(self, box_mask, a_rep, q_rep, obj_reps, question_mask, answer_mask):

        ####################################
        # Perform Q by A attention
        # [batch_size, 4, question_length, answer_length]
        qa_similarity = self.span_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))

        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(
            a_rep.view(a_rep.shape[0], a_rep.shape[1] * a_rep.shape[2], -1),
            obj_reps['obj_reps']).view(a_rep.shape[0], a_rep.shape[1], a_rep.shape[2], obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:, None, None])
        attended_o = torch.einsum('bnao,bod->bnad', (atoo_attention_weights, obj_reps['obj_reps']))

        a_rep, attended_o, attended_q = self.Graph_reasoning(a_rep, attended_o, attended_q)

        reasoning_inp = torch.cat([
            x for x, to_pool in [(a_rep, self.reasoning_use_answer), (
                attended_o, self.reasoning_use_obj), (attended_q, self.reasoning_use_question)] if to_pool
        ], -1)

        if self.rnn_input_dropout is not None:
            reasoning_inp = self.rnn_input_dropout(reasoning_inp)
        reasoning_output = self.reasoning_encoder(reasoning_inp, answer_mask)

        ###########################################
        things_to_pool = torch.cat([
            x for x, to_pool in [(reasoning_output,
                                  self.pool_reasoning), (a_rep, self.pool_answer), (attended_q, self.pool_question)]
            if to_pool
        ], -1)

        pooled_rep = replace_masked_values(things_to_pool.float(), answer_mask[..., None].bool(), -1e7).max(2)[0]

        return pooled_rep


class Graph_reasoning(nn.Module):

    def __init__(self, in_fea):
        super(Graph_reasoning, self).__init__()

        self.hidden_fea = in_fea
        self.hidden_fea_2 = in_fea
        self.final_fea = in_fea
        self.fc_encoder = nn.Linear(in_fea, self.hidden_fea)

        self.fc_o = nn.Linear(self.hidden_fea, self.hidden_fea)
        self.fc_a = nn.Linear(self.hidden_fea, self.hidden_fea)
        self.fc_q = nn.Linear(self.hidden_fea, self.hidden_fea)

        self.fc_o_ = nn.Linear(in_fea + self.hidden_fea, self.hidden_fea)
        self.fc_a_ = nn.Linear(self.hidden_fea, self.hidden_fea)
        self.fc_q_ = nn.Linear(in_fea + self.hidden_fea, self.hidden_fea)

        self.w_s_o = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_a = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_q = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_o_ = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_a_ = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_q_ = nn.Linear(self.hidden_fea, self.hidden_fea_2)

        self.w_g_o = nn.Linear(self.hidden_fea_2, self.final_fea)
        self.w_g_a = nn.Linear(self.hidden_fea_2, self.final_fea)
        self.w_g_q = nn.Linear(self.hidden_fea_2, self.final_fea)

        self.res_w_a = nn.Linear(in_fea * 2, in_fea)
        self.res_w_q = nn.Linear(in_fea * 2, in_fea)
        self.res_w_o = nn.Linear(in_fea * 2, in_fea)

    def forward(self, answer, o_a, q_a):
        bs, num, seq_len, feature = answer.size()
        answer_view = answer.view(bs * num, seq_len, -1)
        o_a_view = o_a.view(bs * num, seq_len, -1)
        q_a_view = q_a.view(bs * num, seq_len, -1)

        encoder_feature = self.fc_encoder(answer_view)
        s_obj = F.softmax(self.fc_o(encoder_feature), -2) * encoder_feature
        s_ans = F.softmax(self.fc_a(encoder_feature), -2) * encoder_feature
        s_que = F.softmax(self.fc_q(encoder_feature), -2) * encoder_feature

        e_obj = self.fc_o_(torch.cat([encoder_feature, o_a_view], -1))
        e_ans = self.fc_a_(encoder_feature)
        e_que = self.fc_q_(torch.cat([encoder_feature, q_a_view], -1))

        A_obj = F.softmax(self.w_g_o(F.relu(self.w_s_o(s_obj) + self.w_s_o_(e_obj))), dim=-2)
        A_ans = F.softmax(self.w_g_a(F.relu(self.w_s_a(s_ans) + self.w_s_a_(e_ans))), dim=-2)
        A_que = F.softmax(self.w_g_q(F.relu(self.w_s_q(s_que) + self.w_s_q_(e_que))), dim=-2)

        a_out = (A_ans * answer_view).view(bs, num, seq_len, feature)
        o_out = (A_obj * o_a_view).view(bs, num, seq_len, feature)
        q_out = (A_que * q_a_view).view(bs, num, seq_len, feature)

        a_out = self.res_w_a(torch.cat([a_out, answer], dim=-1))
        o_out = self.res_w_o(torch.cat([o_out, o_a], dim=-1))
        q_out = self.res_w_q(torch.cat([q_out, q_a], dim=-1))

        return a_out, o_out, q_out
