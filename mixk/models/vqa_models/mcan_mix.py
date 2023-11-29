from ..builder import VQA_MODELS, build_backbone, build_embedding, build_encoder, build_head, build_combine_layer
import torch
from .base_model import BaseModel
from torch import Tensor
from typing import Tuple, Dict, List


def filter_grads(parameters):
    return [param for param in parameters if param.requires_grad]


@VQA_MODELS.register_module()
class MCAN(BaseModel):

    def __init__(self, embedding: 'WordEmbedding and TextEmbedding', encoder: 'ImageFeatureEncoder',
                 backbone: 'TwoBranchEmbedding', combine_model: 'BranchCombineLayer', head: 'TripleLinearHead') -> None:
        """The initialization of MCAN model.

        Args:
            embedding: the embedding of text and word
            encoder: the encoder of image feature
            backbone: backbone networks to process input features
            combine_model: combine the image and text feature
            head: the classifier do the answering job
        """
        super().__init__()
        self.embedding_model = build_embedding(embedding)
        self.encoder_model = build_encoder(encoder)
        self.backbone = build_backbone(backbone)
        self.combine_model = build_combine_layer(combine_model)  # combine text and image
        self.head = build_head(head)  # 包括 classification head， generation head

    def get_optimizer_parameters(self, optimizer_params_lr, training_encoder_lr_multiply):

        combine_layer = self.combine_model
        params = [
            {
                'params': filter_grads(self.embedding_model[0].parameters())
            },
            {
                'params': filter_grads(self.backbone.sga.parameters())
            },
            {
                'params': filter_grads(self.backbone.sga_pool.parameters())
            },
            {
                'params': filter_grads(self.backbone.cbn.parameters()),
                'lr': (optimizer_params_lr * training_encoder_lr_multiply),
            },
            {
                'params': filter_grads(self.embedding_model[-1].parameters())
            },
            {
                'params': filter_grads(combine_layer.parameters())
            },
            {
                'params': filter_grads(self.head.parameters())
            },
            {
                'params': filter_grads(self.encoder_model.parameters())
            },
        ]

        return params

    def process_text_embedding(self, text: Tensor, text_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """conduct text embedding with mask.

        Args:
            text: text in the input data, shaped of (batch_size, 14, 300)
            text_mask: mask used to mask the null place except for real words, shaped of (batch_size, 14)

        Returns:
            Tuple: embedding feature and text embedding vector,shaped of (batch_size, 14, 1024), (batch_size, 2, 1024)
        """
        # Get embedding models
        text_embedding_model = self.embedding_model[-1]
        text_embedding_total, text_embedding_vec = text_embedding_model(text, text_mask)
        return text_embedding_total, text_embedding_vec

    def process_feature_embedding(self,
                                  img_feat: Tensor,
                                  text_embedding_total: Tensor,
                                  text_embedding_vec: Tensor,
                                  text_mask: Tensor,
                                  vextra=None,
                                  batch_size_t=None) -> Tuple[Tensor, Tensor]:
        """embedding image and text to features using self-guided attention and
        features from bottlenecks proposed in MCAN.

        Args:
            img_feat: image features
            text_embedding_total: text embedding total feature
            text_embedding_vec: vectors of text embedding
            text_mask: mask used to mask the non-word places
            vextra: Optinal (not used here)
            batch_size_t: Optinal (not used here)

        Returns:
            Tuple: features from self-guided attention and features from bottlenecks
        """
        image_feature_0 = img_feat
        encoded_feature = self.encoder_model(image_feature_0)
        feature_sga, feature_cbn = self.backbone(encoded_feature, text_embedding_total, text_embedding_vec, None,
                                                 text_mask)

        return feature_sga, feature_cbn

    def _joint_embedding(self, data: Dict, **kwargs: Dict) -> Tensor:
        """the embedding computing of MCAN.

        Args:
            data: a dict containing input data
            **kwargs: other parameters packed into a dict

        Returns:
            Tensor: output containing the embeddings, shaped of (batch_size, 3, 2048)
        """
        batch_data = self.preprocess_data(data)

        img_feat = batch_data['feature']  # (batch_size, 2048, 1024, 1)
        input_ids = batch_data['input_ids']  # (batch_size, 14)
        text_mask = batch_data['input_mask']  # (batch_size, 14)
        # targets = batch_data['answers_scores']

        ques_feat = self.embedding_model[0](input_ids)
        # text_mask = ques_feat.eq(0)
        text_embedding_total, text_embedding_vec = self.process_text_embedding(ques_feat, text_mask)

        feature_sga, feature_cbn = self.process_feature_embedding(img_feat, text_embedding_total,
                                                                  text_embedding_vec[:, 0], text_mask)

        joint_embedding = self.combine_model(feature_sga, feature_cbn, text_embedding_vec[:, 1])
        return joint_embedding

    def forward_train(self, data: List, **kwargs: Dict) -> Dict:
        """the forward training computation of MCAN.

        Args:
            data: A list containing the inputs, using data.keys() to get the specific content
            **kwargs: Other parameters that packed into a Dict

        Returns:
            Dict: the output of the ground truth answers and predicted answers, formed to a dictionary
        """

        batch_data = self.preprocess_data(data)
        joint_embedding = self._joint_embedding(batch_data, **kwargs)

        # targets = batch_data['answers_scores']
        output = self.head.forward(joint_embedding)

        model_output = {}
        model_output['target'] = batch_data['answers_scores']
        model_output['scores'] = output

        return model_output

    def forward_test(self, data: List, **kwargs: Dict) -> Dict:
        """the forward testing computation of MCAN.

        Args:
            data: A list containing the inputs, using data.keys() to get the specific content
            **kwargs: Other parameters that packed into a Dict

        Returns:
            Dict: the output of the ground truth answers and predicted answers, formed to a dictionary
        """
        assert not self.training

        batch_data = self.preprocess_data(data)
        joint_embedding = self._joint_embedding(batch_data, **kwargs)
        output = self.head.forward(joint_embedding)

        model_output = {}
        model_output['target'] = batch_data['answers_scores']
        model_output['scores'] = output

        return model_output

    def preprocess_data(self, batched_inputs: Dict) -> Dict:
        """padding data and turn data into cuda format.

        Args:
            batched_inputs: a Dict containing input data

        Returns:
            Dict: a dict containing input data
        """
        from mixk.engine.organizer import is_by_iter
        if is_by_iter():
            batched_inputs = list2dict(batched_inputs)

        img_feat = batched_inputs['features']
        input_ids = batched_inputs['input_ids']
        input_mask = batched_inputs['input_mask']

        b, c, h, w = img_feat.shape
        feat = img_feat.view(b, c, -1)
        padded_feat = torch.zeros((b, c, 1024), dtype=torch.float)
        padded_feat[:, :, :h * w] = feat
        feat = padded_feat.unsqueeze(-1)
        # feat = feat.squeeze(0)
        feat = feat.cuda()

        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

        batched_inputs['feature'] = feat
        batched_inputs['input_ids'] = input_ids
        batched_inputs['input_mask'] = ~input_mask

        if self.training:
            answers_scores = batched_inputs['answers_scores']
            answers_scores = answers_scores.cuda()
            batched_inputs['answers_scores'] = answers_scores

        return batched_inputs


def list2dict(batched_inputs: List) -> Dict:
    """turn data from list to dict.

    Args:
        batched_inputs: input data

    Returns:
        Dict: a dict containing the input data
    """
    batch_size = len(batched_inputs)
    img_feats = torch.zeros((batch_size, *batched_inputs[0]['features'].shape),
                            dtype=batched_inputs[0]['features'].dtype)
    input_ids = torch.zeros((batch_size, *batched_inputs[0]['input_ids'].shape),
                            dtype=batched_inputs[0]['input_ids'].dtype)
    answers_scores = torch.zeros((batch_size, *batched_inputs[0]['answers_scores'].shape),
                                 dtype=batched_inputs[0]['answers_scores'].dtype)
    input_mask = torch.zeros((batch_size, *batched_inputs[0]['input_mask'].shape),
                             dtype=batched_inputs[0]['input_mask'].dtype)
    question_id = torch.zeros([batch_size], dtype=torch.int32)
    for idx in range(batch_size):
        img_feats[idx] = batched_inputs[idx]['features']
        input_ids[idx] = batched_inputs[idx]['input_ids']
        answers_scores[idx] = batched_inputs[idx]['answers_scores']
        input_mask[idx] = batched_inputs[idx]['input_mask']
        question_id[idx] = batched_inputs[idx]['question_id']

    batch_data = dict()

    batch_data['features'] = img_feats
    batch_data['input_ids'] = input_ids
    batch_data['answers_scores'] = answers_scores
    batch_data['input_mask'] = input_mask
    batch_data['question_id'] = question_id

    return batch_data
