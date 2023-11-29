from copy import deepcopy

import torch
import torch.nn as nn
from transformers.modeling_bert import BertEmbeddings

from ..builder import EMBEDDING


@EMBEDDING.register_module()
class BertVisioLinguisticEmbeddings(BertEmbeddings):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.token_type_embeddings_visual = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def initialize_visual_from_pretrained(self):
        self.token_type_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.token_type_embeddings.weight.data), requires_grad=True)
        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        visual_embeddings=None,
        visual_embeddings_type=None,
        position_embeddings_visual=None,
        image_text_alignment=None,
    ):
        """
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        """

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if visual_embeddings is not None:
            visual_embeddings = self.projection(visual_embeddings)
            token_type_embeddings_visual = self.token_type_embeddings_visual(visual_embeddings_type)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the
                # image feature. -1 is the padding value.
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # position_embeddings_visual
                # = Batch x image_length x alignment length x dim
                position_embeddings_visual = self.position_embeddings(
                    image_text_alignment) * image_text_alignment_mask.to(
                        dtype=next(self.parameters()).dtype).unsqueeze(-1)
                position_embeddings_visual = position_embeddings_visual.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=next(self.parameters()).dtype).sum(2)
                image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid devide by zero error
                position_embeddings_visual = (position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1))

                position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype=torch.long).cuda()

                # When fine-tuning the detector , the image_text_alignment is
                # sometimes padded too long.
                if position_embeddings_visual.size(1) != visual_embeddings.size(1):
                    assert position_embeddings_visual.size(1) >= visual_embeddings.size(1)
                    position_embeddings_visual = position_embeddings_visual[:, :visual_embeddings.size(1), :]

                position_embeddings_visual = (
                    position_embeddings_visual + self.position_embeddings_visual(position_ids_visual))
            else:
                position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype=torch.long).cuda()
                position_embeddings_visual = self.position_embeddings_visual(position_ids_visual)

            v_embeddings = (visual_embeddings + position_embeddings_visual + token_type_embeddings_visual)

            # Concate the two:
            embeddings = torch.cat((embeddings, v_embeddings),
                                   dim=1)  # concat the visual embeddings after the attentions

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@EMBEDDING.register_module()
class BertImageFeatureEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, image_feature, image_location):

        img_embeddings = self.image_embeddings(image_feature)
        loc_embeddings = self.image_location_embeddings(image_location)

        # TODO: we want to make the padding_idx==0, however, with custom initilization,
        # it seems it will have a bias. Let's do masking for now
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
