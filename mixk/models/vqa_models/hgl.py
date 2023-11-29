import torch
import torch.nn as nn
from ..builder import VQA_MODELS
from .r2c import R2C
import torch.nn.functional as F
from torchvision.models import resnet
import torch.utils.model_zoo as model_zoo
from mmcv.ops import RoIAlign  # #import from mmcv


@VQA_MODELS.register_module()
class HGL(R2C):

    def __init__(self, input_dropout, pretrained, average_pool, semantic, final_dim, backbone, head):
        super(HGL, self).__init__(input_dropout, pretrained, average_pool, semantic, final_dim, backbone, head)
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=semantic, final_dim=final_dim)

    def forward_train(self, data, *args, **kwargs):
        # images = data['images']
        # boxes = data['boxes']
        # box_mask = data['box_mask']
        # objects = data['objects']
        # segms = data['segms']
        # question = data['question']
        # question_tags = data['question_tags']
        # question_mask = data['question_mask']
        # answers = data['answers']
        # answer_tags = data['answer_tags']
        # answer_mask = data['answer_mask']

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

        return model_output

    def forward_test(self, data, *args, **kwargs):
        model_output = self.forward_train(data)
        return model_output


class _Context_voted_module(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_Context_voted_module, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class CVM(_Context_voted_module):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(CVM, self).__init__(
            in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)


class RegionCVM(nn.Module):

    def __init__(self, in_channels, grid=[6, 6]):
        super(RegionCVM, self).__init__()
        self.CVM = CVM(in_channels, sub_sample=True, bn_layer=False)
        self.grid = grid

    def forward(self, x):
        batch_size, _, height, width = x.size()

        input_row_list = x.chunk(self.grid[0], dim=2)

        output_row_list = []
        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = row.chunk(self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                grid = self.CVM(grid)
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)
        return output


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
      :param sequence: ['\'sum b, .....] sequence
      :param lengths: [b1, b2, b3...] that sum to '\'sum b
      :return: [len(lengths), maxlen(b), .....] tensor
      """
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    return output


class Flattener(torch.nn.Module):

    def __init__(self):
        """Flattens last 3 dimensions to make it only batch size, -1."""
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleDetector(nn.Module):

    def __init__(self, pretrained=True, average_pool=True, semantic=True, final_dim=1024, layer_fix=True):
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
        self.pre_backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2
        self.cvm_2 = RegionCVM(in_channels=128 * 4, grid=[6, 6])
        self.layer3 = backbone.layer3
        self.cvm_3 = RegionCVM(in_channels=256 * 4, grid=[4, 4])
        self.roi_align = RoIAlign(
            (7, 7) if USE_IMAGENET_PRETRAINED else (14, 14), spatial_scale=1 / 16, sampling_ratio=0)
        if semantic:
            self.mask_dims = 32
            self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
            self.mask_upsample = torch.nn.Conv2d(
                1, self.mask_dims, kernel_size=3, stride=2 if USE_IMAGENET_PRETRAINED else 1, padding=1, bias=True)
        else:
            self.object_embed = None
            self.mask_upsample = None

        self.layer4 = backbone.layer4
        self.cvm_4 = RegionCVM(in_channels=512 * 4, grid=[1, 1])
        after_roi_align = []

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

        for m in self.pre_backbone.modules():
            for p in m.parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.layer2.apply(set_bn_fix)
        self.layer3.apply(set_bn_fix)
        self.layer4.apply(set_bn_fix)
        if layer_fix:
            for m in self.layer2.modules():
                for p in m.parameters():
                    p.requires_grad = False
            for m in self.layer3.modules():
                for p in m.parameters():
                    p.requires_grad = False
            for m in self.layer4.modules():
                for p in m.parameters():
                    p.requires_grad = False

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

        images = self.pre_backbone(images)
        images = self.layer2(images)
        images = self.cvm_2(images)
        images = self.layer3(images)
        images = self.cvm_3(images)
        images = self.layer4(images)
        img_feats = self.cvm_4(images)
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

        # Add some regularization, encouraging the model to keep giving decent enough predictions
        obj_logits = self.regularizing_predictor(post_roialign)
        obj_labels = classes[box_inds[:, 0], box_inds[:, 1]]
        cnn_regularization = F.cross_entropy(obj_logits, obj_labels, reduction='mean')[None]

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
            'cnn_regularization_loss': cnn_regularization
        }
