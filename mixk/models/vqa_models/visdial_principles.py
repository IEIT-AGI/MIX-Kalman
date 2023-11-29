import torch.nn as nn

from ..builder import VQA_MODELS, build_backbone, build_encoder, build_head


@VQA_MODELS.register_module()
class VISDIALPRINCIPLES(nn.Module):

    def __init__(self, vocabulary_len, word_embedding_size, encoder, backbone, head):
        super().__init__()

        self.embedding_model = nn.Embedding(vocabulary_len, word_embedding_size, padding_idx=0)
        self.encoder_model = build_encoder(encoder)
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)  # 包括 classification head， generation head

    def forward(self, data):
        img = data['img_feat']
        ques = data['ques']
        his = data['hist']
        batch_size, rnd, max_his_length = his.size()
        cap = his[:, 0, :]
        ques_len = data['ques_len']
        hist_len = data['hist_len']
        cap_len = hist_len[:, 0]

        ques_embed = self.embedding_model(ques)
        cap_emb = self.embedding_model(cap.contiguous())
        his = his.contiguous().view(-1, max_his_length)
        his_embed = self.embedding_model(his)
        q_output, c_output, his_feat = self.encoder_model(ques_embed, ques_len, cap_emb, cap_len, his_embed, hist_len)
        ques_location = ques_len.view(-1).cpu().numpy() - 1
        ques_encoded = q_output[range(batch_size), ques_location, :]
        cap_location = cap_len.view(-1).cpu().numpy() - 1
        cap_encoded = c_output[range(batch_size), cap_location, :]
        his_feat = his_feat.view(batch_size, rnd, -1)
        fuse_feat = self.backbone(ques_encoded, cap_encoded, his_feat, q_output, c_output, ques_len, cap_len,
                                  ques_embed, cap_emb, img, batch_size)

        scores = self.head(fuse_feat, data)
        return scores
