import math
import random

import cv2
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional


def load_image(img_fn):
    """Load the specified image and return a [H,W,3] Numpy array."""
    return default_loader(img_fn)


def resize_image(image, desired_width=768, desired_height=384, random_pad=False):
    """Resizes an image keeping the aspect ratio mostly unchanged.

    Returns:
    image: the resized image
    window: (x1, y1, x2, y2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [left, top, right, bottom]
    """
    # Default window (x1, y1, x2, y2) and default scale == 1.
    w, h = image.size

    width_scale = desired_width / w
    height_scale = desired_height / h
    scale = min(width_scale, height_scale)

    # Resize image using bilinear interpolation
    if scale != 1:
        image = functional.resize(image, (round(h * scale), round(w * scale)))
    w, h = image.size
    y_pad = desired_height - h
    x_pad = desired_width - w

    top_pad = random.randint(0, y_pad) if random_pad else y_pad // 2
    left_pad = random.randint(0, x_pad) if random_pad else x_pad // 2

    padding = (left_pad, top_pad, x_pad - left_pad, y_pad - top_pad)
    assert all([x >= 0 for x in padding])
    image = functional.pad(image, padding)
    window = [left_pad, top_pad, w + left_pad, h + top_pad]

    return image, window, scale, padding


def to_tensor_and_normalize(image):
    return functional.normalize(functional.to_tensor(image), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def letterbox(img, mask, height, color=(123.7, 116.3, 103.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    if mask is not None:
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)  # resized, no border
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)  # padded square
    return img, mask, ratio, dw, dh


def random_affine(img,
                  mask,
                  targets,
                  degrees=(-10, 10),
                  translate=(.1, .1),
                  scale=(.9, 1.1),
                  shear=(-2, 2),
                  borderValue=(123.7, 116.3, 103.5),
                  all_bbox=None):
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(
        img, M, dsize=(height, height), flags=cv2.INTER_LINEAR, borderValue=borderValue)  # BGR order borderValue
    if mask is not None:
        maskw = cv2.warpPerspective(
            mask, M, dsize=(height, height), flags=cv2.INTER_NEAREST, borderValue=255)  # BGR order borderValue
    else:
        maskw = None

    # Return warped points also
    # if type(targets) == type([1]):
    if isinstance(targets, list):
        # if type(targets[0]) == type([1]):
        if isinstance(targets[0], list):
            targetlist = []
            for bbox in targets:
                targetlist.append(wrap_points(bbox, M, height, a))
            return imw, maskw, targetlist, M
        else:
            targetlist = []
            targetlist.append(wrap_points(targets, M, height, a))
            return imw, maskw, targetlist, M
    elif all_bbox is not None:
        targets = wrap_points(targets, M, height, a)
        for ii in range(all_bbox.shape[0]):
            all_bbox[ii, :] = wrap_points(all_bbox[ii, :], M, height, a)
        return imw, maskw, targets, all_bbox, M
    elif targets is not None:  # previous main
        targets = wrap_points(targets, M, height, a)
        return imw, maskw, targets, M
    else:
        return imw


def wrap_points(targets, M, height, a):
    # n = targets.shape[0]
    # points = targets[:, 1:5].copy()
    points = targets.copy()
    # area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])
    # area0 = (points[2] - points[0]) * (points[3] - points[1])

    # warp points
    xy = np.ones((4, 3))
    xy[:, :2] = np.array(points)[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(1, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

    # apply angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians)))**0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, 1).T

    # reject warped points outside of image
    np.clip(xy, 0, height, out=xy)
    # w = xy[:, 2] - xy[:, 0]
    # h = xy[:, 3] - xy[:, 1]
    # area = w * h
    # ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    # i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    # print(targets, xy)
    # [ 56  36 108 210] [[ 47.80464857  15.6096533  106.30993434 196.71267693]]
    # targets = targets[i]
    # targets[:, 1:5] = xy[i]
    targets = xy[0]
    return targets
