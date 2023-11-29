import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from allennlp.modules import InputVariationalDropout, TimeDistributed
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from mmcv.ops import RoIAlign  # #import from mmcv
from torchvision.models import resnet

from ..builder import VQA_MODELS, build_backbone, build_head
from .base_model import BaseModel


class Flattener(nn.Module):

    def __init__(self):
        """Flattens last 3 dimensions to make it only batch size, -1."""
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


@VQA_MODELS.register_module()
class R2C(BaseModel):

    def __init__(self, input_dropout, pretrained, average_pool, semantic, final_dim, backbone, head):
        super().__init__()

        self.detector = SimpleDetector(
            pretrained=pretrained, average_pool=average_pool, semantic=semantic, final_dim=final_dim)
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None
        self.encoder_model = TimeDistributed(
            PytorchSeq2SeqWrapper(torch.nn.LSTM(1280, 256, batch_first=True, bidirectional=True)))
        self.backbone = build_backbone(backbone)
        # self.combine_model = build_combine_layer(combine_model)  ###combine text and image
        self.head = build_head(head)
        # self.head = torch.nn.Sequential(
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(1536, 1024),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(1024, 1),
        # )  ###包括 classification head， generation head

    def _collect_obj_reps(self, span_tags, object_reps):
        """Collect span-level object representations.

        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span, retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        return self.encoder_model(span_rep, span_mask), retrieved_feats

    def forward_train(self, data, *args, **kwargs):
        images = data['image'].cuda()
        max_num = torch.max(data['max_num']).cuda()
        max_bbox_num = torch.max(data['bbox_num']).cuda()
        boxes = data['boxes'][:, :max_bbox_num, :].cuda()
        # box_mask = data['box_mask'][:, :max_bbox_num, :]
        bbox_mask = torch.all(boxes >= 0, -1).long().cuda()
        objects = data['objects'][:, :max_bbox_num].cuda()
        segms = data['segms'][:, :max_bbox_num, :, :].cuda()
        question = data['questions_embeddings'][:, :, :max_num, :].cuda()
        question_tags = data['questions_obj_tags'][:, :, :max_num].cuda()
        question_mask = data['questions_masks'][:, :, :max_num].cuda()
        answers = data['answers_embeddings'][:, :, :max_num, :].cuda()
        answer_tags = data['answers_obj_tags'][:, :, :max_num].cuda()
        answer_mask = data['answers_masks'][:, :, :max_num].cuda()

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=bbox_mask, classes=objects, segms=segms)
        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        pooled_rep = self.backbone(bbox_mask, a_rep, q_rep, obj_reps, question_mask, answer_mask)
        logits = self.head(pooled_rep)

        model_output = {
            'scores': logits.squeeze(2),
            'target': data['label'].cuda(),
            'obj_scores': obj_reps['obj_logits'],
            'obj_target': obj_reps['obj_labels']
        }

        # loss = F.cross_entropy(logits.squeeze(2), data['label'].cuda())
        #
        # losses = {
        #     'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
        #     'loss': loss
        # }
        return model_output

    def forward_test(self, data, *args, **kwargs):
        model_output = self.forward_train(data)
        return model_output


def _load_resnet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=False)
    if pretrained:
        backbone.load_state_dict(
            model_zoo.load_url('https://s3.us-west-2.amazonaws.com/ai2-rowanz/resnet50-e13db6895d81.th'))
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    return backbone


def _load_resnet_imagenet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=pretrained)
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

    # # Make batchnorm more sensible
    # for submodule in backbone.modules():
    #     if isinstance(submodule, torch.nn.BatchNorm2d):
    #         submodule.momentum = 0.01

    return backbone


def pad_sequence(sequence, lengths):
    """
    :param sequence: [sum b, .....] sequence
    :param lengths: [b1, b2, b3...] that sum to sum b
    :return: [len(lengths), maxlen(b), .....] tensor
    """
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    return output


class SimpleDetector(nn.Module):

    def __init__(self, pretrained=True, average_pool=True, semantic=True, final_dim=1024):
        """
        :param average_pool: whether or not to average pool the representations
        :param pretrained: Whether we need to load from scratch
        :param semantic: Whether or not we want to introduce the mask and the class label early on (default Yes)
        """
        super(SimpleDetector, self).__init__()
        USE_IMAGENET_PRETRAINED = True
        # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
        backbone = _load_resnet_imagenet(pretrained=pretrained) if USE_IMAGENET_PRETRAINED else _load_resnet(
            pretrained=pretrained)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            # backbone.layer4
        )
        self.roi_align = RoIAlign((7, 7) if USE_IMAGENET_PRETRAINED else (14, 14), spatial_scale=1 / 16, sample_num=0)

        if semantic:
            self.mask_dims = 32
            self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
            self.mask_upsample = torch.nn.Conv2d(
                1, self.mask_dims, kernel_size=3, stride=2 if USE_IMAGENET_PRETRAINED else 1, padding=1, bias=True)
        else:
            self.object_embed = None
            self.mask_upsample = None

        after_roi_align = [backbone.layer4]
        self.final_dim = final_dim
        if average_pool:
            after_roi_align += [nn.AvgPool2d(7, stride=1), Flattener()]

        self.after_roi_align = torch.nn.Sequential(*after_roi_align)

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048 + (128 if semantic else 0), final_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.regularizing_predictor = torch.nn.Linear(2048, 81)

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        box_mask: torch.LongTensor,
        classes: torch.Tensor = None,
        segms: torch.Tensor = None,
    ):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :return: object reps [batch_size, max_num_objects, dim]
        """
        # [batch_size, 2048, im_height // 32, im_width // 32
        img_feats = self.backbone(images)
        box_inds = box_mask.nonzero()
        assert box_inds.shape[0] > 0
        rois = torch.cat((
            box_inds[:, 0, None].type(boxes.dtype),
            boxes[box_inds[:, 0], box_inds[:, 1]],
        ), 1)

        # Object class and segmentation representations
        roi_align_res = self.roi_align(img_feats.float(), rois.float())
        if self.mask_upsample is not None:
            assert segms is not None
            segms_indexed = segms[box_inds[:, 0], None, box_inds[:, 1]] - 0.5
            roi_align_res[:, :self.mask_dims] += self.mask_upsample(segms_indexed)

        post_roialign = self.after_roi_align(roi_align_res)
        # post_roialign = torch.rand(16,2048)  #add for test

        # Add some regularization, encouraging the model to keep giving decent enough predictions
        obj_logits = self.regularizing_predictor(post_roialign)
        obj_labels = classes[box_inds[:, 0], box_inds[:, 1]]
        # cnn_regularization = F.cross_entropy(
        #     obj_logits, obj_labels, size_average=True)[None]

        feats_to_downsample = post_roialign if self.object_embed is None else torch.cat(
            (post_roialign, self.object_embed(obj_labels)), -1)
        roi_aligned_feats = self.obj_downsample(feats_to_downsample)

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        obj_reps = pad_sequence(roi_aligned_feats, box_mask.sum(1).tolist())
        return {
            'obj_reps_raw': post_roialign,
            'obj_reps': obj_reps,
            'obj_logits': obj_logits,
            'obj_labels': obj_labels,
            # 'cnn_regularization_loss': cnn_regularization,
        }
