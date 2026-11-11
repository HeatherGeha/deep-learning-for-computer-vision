"""PH2 skin lesion dataset.

Layout: <root>/IMD002/IMD002_Dermoscopic_Image/IMD002.bmp  (image)
        <root>/IMD002/IMD002_lesion/IMD002_lesion.bmp      (lesion mask)
The IMD*_roi folders are not used. PH2 has no pre-assigned splits.

Run `python datasets_PH2.py` on the HPC to check the data and print its stats.
"""
from glob import glob
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seg_common import random_flips, split_ids, transform_sample

PH2_ROOT = '/dtu/datasets1/02516/PH2_Dataset_images'
#change root for heather to work locally
#PH2_ROOT = r'C:/Users/heath/Desktop/Heather Python/DTU/Deep Learning in Computer Vision/PH2_Dataset_images'
# About the 3:4 shape of the original images, and divisible by 32 for the U-Net
PH2_SIZE = (288, 384)  # (height, width)
PH2_N_VAL = 30
PH2_N_TEST = 30
CLICK_STRATEGIES = ('random', 'spread')
# Size of the oval that 'spread' keeps negative clicks in (1.0 touches the image frame)
VIEW_OVAL_SCALE = 0.85


class PH2Dataset(torch.utils.data.Dataset):
    """Returns (image [3, H, W] in [0, 1], mask [1, H, W] in {0, 1})."""

    def __init__(self, root_dir=PH2_ROOT, split='train', size=PH2_SIZE, augment=None):
        ids = [os.path.basename(p) for p in glob(os.path.join(root_dir, 'IMD*'))]
        if not ids:
            raise ValueError(f'No IMD* folders found in {root_dir}')

        self.ids = split_ids(ids, PH2_N_VAL, PH2_N_TEST)[split]
        self.root_dir = root_dir
        self.size = size
        self.augment = (split == 'train') if augment is None else augment

    def __len__(self):
        return len(self.ids)

    def paths(self, idx):
        sample_id = self.ids[idx]
        folder = os.path.join(self.root_dir, sample_id)
        return (
            os.path.join(folder, f'{sample_id}_Dermoscopic_Image', f'{sample_id}.bmp'),
            os.path.join(folder, f'{sample_id}_lesion', f'{sample_id}_lesion.bmp'),
        )

    def __getitem__(self, idx):
        image_path, mask_path = self.paths(idx)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image, (mask,) = transform_sample(image, [mask], self.size, self.augment)
        return image, mask


def GetPH2Loaders(root_dir=PH2_ROOT, batch_size=8, num_workers=2, size=PH2_SIZE):
    """Returns test, train, val loaders (same order as Project 2)."""
    train_set = PH2Dataset(root_dir, split='train', size=size)
    val_set = PH2Dataset(root_dir, split='val', size=size)
    test_set = PH2Dataset(root_dir, split='test', size=size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader, train_loader, val_loader


def _erode(region, margin):
    """Shrink a boolean [H, W] region by margin pixels. Image borders don't count as edges."""
    outside = (~region)[None, None].float()
    return F.max_pool2d(outside, 2 * margin + 1, stride=1, padding=margin)[0, 0] == 0


def _view_oval(shape):
    """Boolean [H, W] oval that leaves out the dark corners from the dermoscope."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, shape[0]), torch.linspace(-1, 1, shape[1]), indexing='ij')
    return ys ** 2 + xs ** 2 <= VIEW_OVAL_SCALE ** 2


def _pick_spread(coords, n, rng, n_candidates=5):
    """Best-candidate sampling: each click is the farthest of a few random options.

    More candidates spread the clicks further, but push them to the region's edge.
    """
    first = rng.integers(len(coords))
    chosen = [coords[first]]
    coords = np.delete(coords, first, axis=0)
    while len(chosen) < n:
        candidates = rng.choice(len(coords), size=min(n_candidates, len(coords)), replace=False)
        distances = ((coords[candidates][:, None] - np.array(chosen)[None]) ** 2).sum(-1).min(1)
        best = candidates[distances.argmax()]
        chosen.append(coords[best])
        coords = np.delete(coords, best, axis=0)
    return np.array(chosen)


def _sample_region(region, n, strategy, margin, rng, allowed=None):
    """Returns up to n (y, x) pixels from a boolean [H, W] region."""
    if strategy == 'spread':
        preferred = _erode(region, margin)
        if allowed is not None:
            preferred = preferred & allowed
        if preferred.sum() >= n:  # regions too small for this keep all of their pixels
            region = preferred

    coords = np.argwhere(region.numpy())
    n = min(n, len(coords))
    if n == 0:
        return coords[:0]
    if strategy == 'spread':
        return _pick_spread(coords, n, rng)
    return coords[rng.choice(len(coords), size=n, replace=False)]


def sample_clicks(mask, n_pos, n_neg, strategy='random', margin=10, seed=(0,)):
    """Simulate an annotator clicking on the lesion and on the background.

    mask: full mask [1, H, W] in {0, 1}, used only to place the clicks.
    strategy: 'random' samples uniformly inside and outside the lesion.
              'spread' stays margin pixels away from the lesion edge, where a
              person would avoid clicking, and spreads the clicks out. Its
              negative clicks stay inside an oval, off the dark image corners.

    Returns a click map [1, H, W]: 1 = positive click, 0 = negative click,
    -1 = unlabelled (to be ignored by the loss).

    Positive and negative clicks use separate random streams, so changing n_pos
    leaves the negative clicks unchanged, and vice versa.
    """
    if strategy not in CLICK_STRATEGIES:
        raise ValueError(f'Unknown strategy {strategy!r}, expected one of {CLICK_STRATEGIES}')

    lesion = mask[0] > 0.5
    clicks = torch.full_like(mask, -1.0)
    oval = _view_oval(lesion.shape)
    for label, region, n, allowed in ((1, lesion, n_pos, None), (0, ~lesion, n_neg, oval)):
        rng = np.random.default_rng([*seed, label])
        picked = _sample_region(region, n, strategy, margin, rng, allowed)
        clicks[0, picked[:, 0], picked[:, 1]] = float(label)
    return clicks


class PH2ClickDataset(PH2Dataset):
    """PH2 training images with simulated clicks instead of full masks.

    Returns (image [3, H, W] in [0, 1], click map [1, H, W] in {-1, 0, 1}).

    Clicks are placed after resizing and stay the same for an image across
    epochs, like real annotations. Flips are applied to the image and clicks.
    """

    def __init__(self, root_dir=PH2_ROOT, n_pos=5, n_neg=5, strategy='random', margin=10,
                 size=PH2_SIZE, flip=True, click_seed=0):
        super().__init__(root_dir, split='train', size=size, augment=False)
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.strategy = strategy
        self.margin = margin
        self.flip = flip
        self.click_seed = click_seed

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        seed = (self.click_seed, int(self.ids[idx][len('IMD'):]))
        clicks = sample_clicks(mask, self.n_pos, self.n_neg, self.strategy, self.margin, seed)
        if self.flip:
            image, clicks = random_flips([image, clicks])
        return image, clicks


def GetPH2ClickLoaders(root_dir=PH2_ROOT, n_pos=5, n_neg=5, strategy='random', margin=10,
                       batch_size=8, num_workers=2, size=PH2_SIZE, click_seed=0):
    """Returns test, train, val loaders: clicks for train, full masks for val and test."""
    test_loader, _, val_loader = GetPH2Loaders(root_dir, batch_size, num_workers, size)
    train_set = PH2ClickDataset(root_dir, n_pos, n_neg, strategy, margin, size, click_seed=click_seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return test_loader, train_loader, val_loader


if __name__ == '__main__':
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else PH2_ROOT
    for split in ('train', 'val', 'test'):
        dataset = PH2Dataset(root_dir, split=split, augment=False)
        sizes = set()
        foreground = []
        for i in range(len(dataset)):
            sizes.add(Image.open(dataset.paths(i)[0]).size)
            image, mask = dataset[i]
            assert set(mask.unique().tolist()) <= {0.0, 1.0}, dataset.ids[i]
            foreground.append(mask.mean().item())
        print(f'{split}: {len(dataset)} images, tensors {tuple(image.shape)} / {tuple(mask.shape)}, '
              f'mean lesion fraction {sum(foreground) / len(foreground):.3f}')
        print(f'  original (W, H) sizes: {sorted(sizes)}')
        print(f'  ids: {sorted(dataset.ids)}')

    for strategy in CLICK_STRATEGIES:
        dataset = PH2ClickDataset(root_dir, n_pos=5, n_neg=5, strategy=strategy, flip=False)
        n_pos, n_neg, outside_oval = 0, 0, 0
        for i in range(len(dataset)):
            _, clicks = dataset[i]
            _, mask = PH2Dataset.__getitem__(dataset, i)
            assert (mask[clicks == 1] == 1).all() and (mask[clicks == 0] == 0).all(), dataset.ids[i]
            n_pos += (clicks == 1).sum().item()
            n_neg += (clicks == 0).sum().item()
            outside_oval += ((clicks[0] == 0) & ~_view_oval(mask.shape[1:])).sum().item()
        print(f'clicks ({strategy}): {len(dataset)} images, {n_pos} positive, {n_neg} negative, '
              f'all on the correct side of the mask, {outside_oval} negative outside the oval')
