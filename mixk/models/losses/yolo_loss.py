import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..builder import LOSSES
from .base_loss import BaseLoss


@LOSSES.register_module()
class YOLOLoss(BaseLoss):
    loss_name = 'yolo_loss'

    def __init__(self):
        super().__init__()
        anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
        anchors = [float(x) for x in anchors.split(',')]
        self.anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]

    def forward(self, model_output):
        predict_anchor, target_bbox = model_output['scores'], model_output['target']
        losses = YOLOLoss.compute_loss(predict_anchor, target_bbox, self.anchors_full)
        return losses

    @staticmethod
    def calculate_loss(input, target, gi, gj, best_n_list, w_coord=5., w_neg=1. / 5, size_average=True):
        mseloss = torch.nn.MSELoss(size_average=True)
        celoss = torch.nn.CrossEntropyLoss(size_average=True)
        batch = input[0].size(0)

        pred_bbox = Variable(torch.zeros(batch, 4).cuda())
        gt_bbox = Variable(torch.zeros(batch, 4).cuda())
        for ii in range(batch):
            pred_bbox[ii, 0:2] = F.sigmoid(input[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 0:2, gj[ii], gi[ii]])
            pred_bbox[ii, 2:4] = input[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 2:4, gj[ii], gi[ii]]
            gt_bbox[ii, :] = target[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, :4, gj[ii], gi[ii]]
        loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
        loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
        loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
        loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])

        pred_conf_list, gt_conf_list = [], []
        for scale_ii in range(len(input)):
            pred_conf_list.append(input[scale_ii][:, :, 4, :, :].contiguous().view(batch, -1))
            gt_conf_list.append(target[scale_ii][:, :, 4, :, :].contiguous().view(batch, -1))
        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
        return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf

    @staticmethod
    def compute_loss(pred_anchor_list, bbox, anchors_full):
        gt_param, gi, gj, best_n_list = YOLOLoss.build_target(bbox, pred_anchor_list, anchors_full)
        for ii in range(len(pred_anchor_list)):
            pred_anchor_list[ii] = pred_anchor_list[ii].view(pred_anchor_list[ii].size(0), 3, 5,
                                                             pred_anchor_list[ii].size(2), pred_anchor_list[ii].size(3))
        # loss
        loss = YOLOLoss.calculate_loss(pred_anchor_list, gt_param, gi, gj, best_n_list)
        return loss
        # losses = {'loss': loss}

        # return losses

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True):
        """Returns the IoU of two bounding boxes."""
        if x1y1x2y2:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        else:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        # get the coordinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
        # Union Area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        # print(box1, box1.shape)
        # print(box2, box2.shape)
        return inter_area / (b1_area + b2_area - inter_area + 1e-16)

    @staticmethod
    def build_target(raw_coord, pred, anchors_full):
        size = 256
        anchor_imsize = 416
        coord_list, bbox_list = [], []
        for scale_ii in range(len(pred)):
            coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
            batch, grid = raw_coord.size(0), size // (32 // (2**scale_ii))
            coord[:, 0] = (raw_coord[:, 0] + raw_coord[:, 2]) / (2 * size)
            coord[:, 1] = (raw_coord[:, 1] + raw_coord[:, 3]) / (2 * size)
            coord[:, 2] = (raw_coord[:, 2] - raw_coord[:, 0]) / size
            coord[:, 3] = (raw_coord[:, 3] - raw_coord[:, 1]) / size
            coord = coord * grid
            coord_list.append(coord)
            bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

        best_n_list, best_gi, best_gj = [], [], []

        for ii in range(batch):
            anch_ious = []
            for scale_ii in range(len(pred)):
                batch, grid = raw_coord.size(0), size // (32 // (2**scale_ii))
                gi = coord_list[scale_ii][ii, 0].long()
                gj = coord_list[scale_ii][ii, 1].long()
                tx = coord_list[scale_ii][ii, 0] - gi.float()
                ty = coord_list[scale_ii][ii, 1] - gj.float()

                gw = coord_list[scale_ii][ii, 2]
                gh = coord_list[scale_ii][ii, 3]

                anchor_idxs = [x + 3 * scale_ii for x in [0, 1, 2]]
                anchors = [anchors_full[i] for i in anchor_idxs]
                scaled_anchors = [(x[0] / (anchor_imsize / grid), x[1] / (anchor_imsize / grid)) for x in anchors]

                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh], dtype=np.float32)).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(
                    np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious += list(YOLOLoss.bbox_iou(gt_box, anchor_shapes))
            # Find the best matching anchor box
            best_n = np.argmax(np.array(anch_ious))
            best_scale = best_n // 3

            batch, grid = raw_coord.size(0), size // (32 / (2**best_scale))
            anchor_idxs = [x + 3 * best_scale for x in [0, 1, 2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (anchor_imsize / grid), x[1] / (anchor_imsize / grid)) for x in anchors]

            gi = coord_list[best_scale][ii, 0].long()
            gj = coord_list[best_scale][ii, 1].long()
            tx = coord_list[best_scale][ii, 0] - gi.float()
            ty = coord_list[best_scale][ii, 1] - gj.float()
            gw = coord_list[best_scale][ii, 2]
            gh = coord_list[best_scale][ii, 3]
            tw = torch.log(gw / scaled_anchors[best_n % 3][0] + 1e-16)
            th = torch.log(gh / scaled_anchors[best_n % 3][1] + 1e-16)

            bbox_list[best_scale][ii, best_n % 3, :, gj,
                                  gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
            best_n_list.append(int(best_n))
            best_gi.append(gi)
            best_gj.append(gj)

        for ii in range(len(bbox_list)):
            bbox_list[ii] = Variable(bbox_list[ii].cuda())
        return bbox_list, best_gi, best_gj, best_n_list


@LOSSES.register_module()
class YOLOLossV2(BaseLoss):
    loss_name = 'yolo_loss_v2'

    def __init__(self):
        super().__init__()
        anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
        anchors = [float(x) for x in anchors.split(',')]
        self.anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]

    def forward(self, model_output):
        predict_anchor, target_bbox = model_output['scores'], model_output['target']
        losses = YOLOLossV2.compute_loss(predict_anchor, target_bbox, self.anchors_full)
        return losses

    @staticmethod
    def calculate_loss(input, target, gi, gj, best_n_list, w_coord=5., w_neg=1. / 5, size_average=True):
        mseloss = torch.nn.MSELoss(size_average=True)
        celoss = torch.nn.CrossEntropyLoss(size_average=True)
        batch = input.size(0)

        pred_bbox = Variable(torch.zeros(batch, 4).cuda())
        gt_bbox = Variable(torch.zeros(batch, 4).cuda())
        for ii in range(batch):
            pred_bbox[ii, 0:2] = F.sigmoid(input[ii, best_n_list[ii], 0:2, gj[ii], gi[ii]])
            pred_bbox[ii, 2:4] = input[ii, best_n_list[ii], 2:4, gj[ii], gi[ii]]
            gt_bbox[ii, :] = target[ii, best_n_list[ii], :4, gj[ii], gi[ii]]
        loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
        loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
        loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
        loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])

        pred_conf_list, gt_conf_list = [], []
        pred_conf_list.append(input[:, :, 4, :, :].contiguous().view(batch, -1))
        gt_conf_list.append(target[:, :, 4, :, :].contiguous().view(batch, -1))
        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
        return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf

    @staticmethod
    def compute_loss(pred_anchor_list, bbox, anchors_full):

        loss = 0.
        for pred_anchor in pred_anchor_list:
            # convert gt box to center+offset format
            gt_param, gi, gj, best_n_list = YOLOLossV2.build_target(bbox, pred_anchor, anchors_full)
            # flatten anchor dim at each scale
            pred_anchor = pred_anchor.view(pred_anchor.size(0), 9, 5, pred_anchor.size(2), pred_anchor.size(3))
            # loss
            loss += YOLOLossV2.calculate_loss(pred_anchor, gt_param, gi, gj, best_n_list)

        return loss

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True):
        """Returns the IoU of two bounding boxes."""
        if x1y1x2y2:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        else:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        # get the coordinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
        # Union Area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        # print(box1, box1.shape)
        # print(box2, box2.shape)
        return inter_area / (b1_area + b2_area - inter_area + 1e-16)

    @staticmethod
    def build_target(raw_coord, pred, anchors_full):
        size = 256
        gsize = 8
        coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
        batch, grid = raw_coord.size(0), size // gsize
        coord[:, 0] = (raw_coord[:, 0] + raw_coord[:, 2]) / (2 * size)
        coord[:, 1] = (raw_coord[:, 1] + raw_coord[:, 3]) / (2 * size)
        coord[:, 2] = (raw_coord[:, 2] - raw_coord[:, 0]) / size
        coord[:, 3] = (raw_coord[:, 3] - raw_coord[:, 1]) / size
        coord = coord * grid
        bbox = torch.zeros(coord.size(0), 9, 5, grid, grid)

        best_n_list, best_gi, best_gj = [], [], []

        for ii in range(batch):
            batch, grid = raw_coord.size(0), size // gsize
            gi = coord[ii, 0].long()
            gj = coord[ii, 1].long()
            tx = coord[ii, 0] - gi.float()
            ty = coord[ii, 1] - gj.float()
            gw = coord[ii, 2]
            gh = coord[ii, 3]

            anchor_idxs = range(9)
            anchor_imsize = 416
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (anchor_imsize / grid), x[1] / (anchor_imsize / grid)) for x in anchors]

            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh], dtype=np.float32)).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(
                np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
            # Calculate iou between gt and anchor shapes
            # anch_ious = list(bbox_iou(gt_box, anchor_shapes))
            anch_ious = list(YOLOLossV2.bbox_iou(gt_box, anchor_shapes, x1y1x2y2=False))
            # Find the best matching anchor box
            best_n = np.argmax(np.array(anch_ious))

            tw = torch.log(gw / scaled_anchors[best_n][0] + 1e-16)
            th = torch.log(gh / scaled_anchors[best_n][1] + 1e-16)

            bbox[ii, best_n, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
            best_n_list.append(int(best_n))
            best_gi.append(gi)
            best_gj.append(gj)

        bbox = Variable(bbox.cuda())
        return bbox, best_gi, best_gj, best_n_list
