# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms():

    min_size = 800
    max_size = 1333
    flip_horizontal_prob = 0.0
    flip_vertical_prob = 0.0
    brightness = 0.0
    contrast = 0.0
    saturation = 0.0
    hue = 0.0

    to_bgr255 = True
    normalize_transform = T.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.], to_bgr255=to_bgr255)
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose([
        color_jitter,
        T.Resize(min_size, max_size),
        T.RandomHorizontalFlip(flip_horizontal_prob),
        T.RandomVerticalFlip(flip_vertical_prob),
        T.ToTensor(),
        normalize_transform,
    ])
    return transform
