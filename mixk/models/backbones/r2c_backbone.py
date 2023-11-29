import torch
import torch.nn as nn
from allennlp.modules import InputVariationalDropout, TimeDistributed
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import masked_softmax, replace_masked_values

from ..builder import BACKBONES


@BACKBONES.register_module()
class R2C_BACKBONE(nn.Module):
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

    def __init__(self,
                 pretrained=True,
                 average_pool=True,
                 semantic=True,
                 final_dim=512,
                 input_dropout=0.3,
                 reasoning_use_obj=True,
                 reasoning_use_answer=True,
                 reasoning_use_question=True,
                 pool_reasoning=True,
                 pool_answer=True,
                 pool_question=True):
        super().__init__()

        # self.detector = SimpleDetector(pretrained=pretrained,
        #   average_pool=average_pool, semantic=semantic, final_dim=final_dim)
        self.reasoning_encoder = TimeDistributed(
            PytorchSeq2SeqWrapper(torch.nn.LSTM(1536, 256, num_layers=2, batch_first=True, bidirectional=True)))
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None
        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=final_dim,
            matrix_2_dim=final_dim,
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=final_dim,
            matrix_2_dim=final_dim,
        )
        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question

        InitializerApplicator(self)

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
