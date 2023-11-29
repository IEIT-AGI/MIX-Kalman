import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder


@VQA_MODELS.register_module()
class CAGRAPH(nn.Module):

    def __init__(self, embedding, encoder, backbone, batch_size):
        super(CAGRAPH, self).__init__()
        self.embedding_model = build_embedding(embedding)
        self.encoder = build_encoder(encoder)
        self.backbone = build_backbone(backbone)  # 包括two_branch_embedding(MCAN), BERT

        self.ques_hidden = self.backbone.init_hidden(batch_size)
        self.his_hidden = self.backbone.init_hidden(batch_size)
        self.real_hidden = self.backbone.init_hidden(batch_size)
        self.wrong_hidden = self.backbone.init_hidden(batch_size)  # question dataset
        self.nhid = 512
        self.img_embed = nn.Linear(2048, self.nhid)

    def forward_rnd(self, ques_emb, his_emb, img, ques_hidden, his_hidden, rnd):

        # b = img.size(0)
        # L = ques_emb.size(0)
        r_feat = img.contiguous().view(-1, 36, 2048)
        r_feat = r_feat / (r_feat.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(r_feat)
        rcnn_feat = F.tanh(self.img_embed(r_feat))

        ques_feat1, ques_hidden = self.encoder[0](ques_emb, ques_hidden)
        # ques_feat = ques_feat1[-1]

        his_feat, his_hidden = self.encoder[1](his_emb, his_hidden)
        his_feat = his_feat[-1]
        final_feat = self.backbone(ques_feat1, his_feat, rcnn_feat, ques_emb, rnd)

        return final_feat, ques_hidden

    def forward(self, data):

        image = data['image']
        history = data['history']
        question = data['question']
        # answer = data['answer']
        answerT = data['answerT']
        # answerLen = data['answerLen']
        answerIdx = data['answerIdx']
        # questionL = data['questionL']
        opt_answerT = data['opt_answerT']
        # opt_answerLen = data['opt_answerLen']
        opt_answerIdx = data['opt_answerIdx']

        batch_size = question.size(0)

        image = image.view(-1, 36, 2048)  # image : batchx36x2048
        # img_input.data.resize_(image.size()).copy_(image)
        img_input = image

        featD_all = []
        real_feat_all = []
        wrong_feat_all = []
        batch_wrong_feat_all = []

        for rnd in range(10):
            self.embedding_model.zero_grad()
            self.encoder.zero_grad()
            self.backbone.zero_grad()

            # get the corresponding round QA and history.
            ques = question[:, rnd, :].t()
            his = history[:, :rnd + 1, :].clone().view(-1, 24).t()  # his_length=24

            # ans = answer[:, rnd, :].t()
            tans = answerT[:, rnd, :].t()
            wrong_ans = opt_answerT[:, rnd, :].clone().view(-1, 9).t()  # ans_length=9

            # real_len = answerLen[:, rnd]
            # wrong_len = opt_answerLen[:, rnd, :].clone().view(-1)

            # ques_input.resize_(ques.size()).copy_(ques)
            ques_input = ques

            # his_input.resize_(his.size()).copy_(his)
            his_input = his

            # ans_input.resize_(ans.size()).copy_(ans)
            # ans_target.resize_(tans.size()).copy_(tans)
            # wrong_ans_input.resize_(wrong_ans.size()).copy_(wrong_ans)

            # ans_input = ans
            ans_target = tans
            wrong_ans_input = wrong_ans

            # sample in-batch negative index

            batch_sample_idx = Variable(torch.LongTensor(batch_size))
            batch_sample_idx.resize_(batch_size, 30).zero_()  # neg_batch_sample=30
            sample_batch_neg(answerIdx[:, rnd], opt_answerIdx[:, rnd, :], batch_sample_idx, 30)  # neg_batch_sample=30

            ques_emb = self.embedding_model(ques_input, format='index')
            his_emb = self.embedding_model(his_input, format='index')

            self.ques_hidden = repackage_hidden(self.ques_hidden, batch_size)
            self.his_hidden = repackage_hidden(self.his_hidden, his_input.size(1))

            featD, ques_hidden = self.forward_rnd(ques_emb, his_emb, img_input, self.ques_hidden, self.his_hidden,
                                                  rnd + 1)

            self.ques_hidden = tuple([x.detach() for x in ques_hidden])

            ans_real_emb = self.embedding_model(ans_target, format='index')
            ans_wrong_emb = self.embedding_model(wrong_ans_input, format='index')
            self.real_hidden = repackage_hidden(self.real_hidden, batch_size)
            self.wrong_hidden = repackage_hidden(self.wrong_hidden, ans_wrong_emb.size(1))
            real_feat = self.encoder[-1](ans_real_emb, ans_target, self.real_hidden)
            wrong_feat = self.encoder[-1](ans_wrong_emb, wrong_ans_input, self.wrong_hidden)

            batch_wrong_feat = wrong_feat.index_select(0, batch_sample_idx.view(-1))
            wrong_feat = wrong_feat.view(batch_size, -1, ques_emb.size()[-1])
            batch_wrong_feat = batch_wrong_feat.view(batch_size, -1, ques_emb.size()[-1])

            featD_all.append(featD)
            real_feat_all.append(real_feat)
            wrong_feat_all.append(wrong_feat)
            batch_wrong_feat_all.append(batch_wrong_feat)

        return featD_all, real_feat_all, wrong_feat_all, batch_wrong_feat_all


def sample_batch_neg(answerIdx, negAnswerIdx, sample_idx, num_sample):
    """
    input:
    answerIdx: batch_size
    negAnswerIdx: batch_size x opt.negative_sample

    output:
    sample_idx = batch_size x num_sample
    """

    batch_size = answerIdx.size(0)
    num_neg = negAnswerIdx.size(0) * negAnswerIdx.size(1)
    negAnswerIdx = negAnswerIdx.clone().view(-1)
    for b in range(batch_size):
        gt_idx = answerIdx[b]

        for n in range(num_sample):
            while True:
                rand = int(random.random() * num_neg)

                neg_idx = negAnswerIdx[rand]
                if gt_idx != neg_idx:
                    sample_idx.data[b, n] = rand
                    break


def repackage_hidden(h, batch_size):
    """Wraps hidden states in new Variables, to detach them from their
    history."""
    # if type(h) == Variable:
    if isinstance(h, Variable):
        return Variable(h.resize_(h.size(0), batch_size, h.size(2)).zero_())
    else:
        return tuple(repackage_hidden(v, batch_size) for v in h)
