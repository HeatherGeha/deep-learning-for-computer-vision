"""Helpers shared by datasets_PH2.py and datasets_DRIVE.py."""
import random

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def split_ids(ids, n_val, n_test, seed=42):
    """Seeded split: the same ids and seed give the same split on every machine.

    Part 2 must use the same splits as Part 1, so keep the default seed.
    """
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    return {
        'val': ids[:n_val],
        'test': ids[n_val:n_val + n_test],
        'train': ids[n_val + n_test:],
    }


def random_flips(tensors):
    """Apply the same random horizontal and vertical flips to every tensor."""
    if torch.rand(1) < 0.5:
        tensors = [TF.hflip(t) for t in tensors]
    if torch.rand(1) < 0.5:
        tensors = [TF.vflip(t) for t in tensors]
    return tensors


def transform_sample(image, masks, size, augment):
    """Apply one set of random transforms to an image and all of its masks.

    T.Compose would draw separate random flips for each, misaligning them.
    Masks use nearest-neighbour resizing so they stay binary.

    Returns the image as a FloatTensor in [0, 1] with shape [3, H, W], and a
    list of masks as FloatTensors in {0, 1} with shape [1, H, W].
    """
    image = TF.resize(image, size, InterpolationMode.BILINEAR, antialias=True)
    masks = [TF.resize(m, size, InterpolationMode.NEAREST) for m in masks]

    image = TF.to_tensor(image)
    masks = [(TF.to_tensor(m) > 0.5).float() for m in masks]

    if augment:
        image, *masks = random_flips([image, *masks])
    return image, masks
