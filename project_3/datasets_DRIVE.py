"""DRIVE retinal vessel dataset.

Layout: <root>/training/images/21_training.tif       (image)
        <root>/training/1st_manual/21_manual1.gif    (vessel mask: the target)
        <root>/training/mask/21_training_mask.gif    (field-of-view mask: NOT the target)

DRIVE/test has images and field-of-view masks but no vessel masks, so it cannot
be used for evaluation. Train, val and test are all made from the 20 labelled
images in DRIVE/training.

Run `python datasets_DRIVE.py` on the HPC to check the data and print its stats.
"""
from glob import glob
import os

from PIL import Image
import torch
from torch.utils.data import DataLoader

from seg_common import split_ids, transform_sample

import torchvision.transforms.functional as TF
import random


DRIVE_ROOT = '/dtu/datasets1/02516/DRIVE'
# Close to the original 584 x 565, and divisible by 32 for the U-Net
DRIVE_SIZE = (512, 512)  # (height, width)
DRIVE_N_VAL = 4
DRIVE_N_TEST = 4


class DRIVEDataset(torch.utils.data.Dataset):
    """Returns (image [3, H, W] in [0, 1], vessel mask [1, H, W] in {0, 1}).

    With return_fov=True, also returns the field-of-view mask [1, H, W], so
    metrics can be computed inside the retina only.
    """

    def __init__(self, root_dir=DRIVE_ROOT, split='train', size=DRIVE_SIZE, augment=None, return_fov=False):
        self.training_dir = os.path.join(root_dir, 'training')
        paths = glob(os.path.join(self.training_dir, 'images', '*_training.tif'))
        ids = [os.path.basename(p).split('_')[0] for p in paths]
        if not ids:
            raise ValueError(f'No *_training.tif images found in {self.training_dir}/images')

        self.ids = split_ids(ids, DRIVE_N_VAL, DRIVE_N_TEST)[split]
        self.size = size
        self.augment = (split == 'train') if augment is None else augment
        self.return_fov = return_fov

    def __len__(self):
        return len(self.ids)

    def paths(self, idx):
        sample_id = self.ids[idx]
        return (
            os.path.join(self.training_dir, 'images', f'{sample_id}_training.tif'),
            os.path.join(self.training_dir, '1st_manual', f'{sample_id}_manual1.gif'),
            os.path.join(self.training_dir, 'mask', f'{sample_id}_training_mask.gif'),
        )

    def __getitem__(self, idx):
        image_path, vessel_path, fov_path = self.paths(idx)
        image = Image.open(image_path).convert('RGB')
        vessels = Image.open(vessel_path).convert('L')
        fov = Image.open(fov_path).convert('L')
        image, (vessels, fov) = transform_sample(image, [vessels, fov], self.size, self.augment)
        if self.return_fov:
            return image, vessels, fov
        return image, vessels


def GetDRIVELoaders(root_dir=DRIVE_ROOT, batch_size=2, num_workers=2, size=DRIVE_SIZE, return_fov=False):
    """Returns test, train, val loaders (same order as Project 2)."""
    train_set = DRIVEDataset(root_dir, split='train', size=size, return_fov=return_fov)
    val_set = DRIVEDataset(root_dir, split='val', size=size, return_fov=return_fov)
    test_set = DRIVEDataset(root_dir, split='test', size=size, return_fov=return_fov)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader, train_loader, val_loader


if __name__ == '__main__':
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else DRIVE_ROOT
    for split in ('train', 'val', 'test'):
        dataset = DRIVEDataset(root_dir, split=split, augment=False, return_fov=True)
        sizes = set()
        vessel_in_fov = []
        for i in range(len(dataset)):
            sizes.add(Image.open(dataset.paths(i)[0]).size)
            image, vessels, fov = dataset[i]
            for mask in (vessels, fov):
                assert set(mask.unique().tolist()) <= {0.0, 1.0}, dataset.ids[i]
            vessel_in_fov.append((vessels * fov).sum().item() / fov.sum().item())
        print(f'{split}: {len(dataset)} images, tensors {tuple(image.shape)} / {tuple(vessels.shape)}, '
              f'mean vessel fraction inside FOV {sum(vessel_in_fov) / len(vessel_in_fov):.3f}')
        print(f'  original (W, H) sizes: {sorted(sizes)}')
        print(f'  ids: {sorted(dataset.ids)}')

class AugmentedDRIVEDataset(DRIVEDataset):
    """
    Extension of DRIVEDataset applying joint spatial rotations, flips,
    and photometric noise.
    """
    def __init__(self, root_dir=DRIVE_ROOT, split='train', size=DRIVE_SIZE, return_fov=False):
        super().__init__(root_dir=root_dir, split=split, size=size, return_fov=return_fov)
        self.split = split  

    def __getitem__(self, idx):
        image, vessels = super().__getitem__(idx)
        
        # Only apply during training
        if self.split == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
                vessels = TF.hflip(vessels)
            if random.random() > 0.5:
                image = TF.vflip(image)
                vessels = TF.vflip(vessels)
            if random.random() > 0.5:
                angle = random.uniform(-45, 45)
                image = TF.rotate(image, angle)
                vessels = TF.rotate(vessels, angle)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)

        return image, vessels

def GetAugmentedDRIVELoaders(root_dir=DRIVE_ROOT, batch_size=2, num_workers=0, size=DRIVE_SIZE):
    """
    DataLoader factory function for Extended U-Net (Hybrid Loss + Data Augmentations).
    """
    train_set = AugmentedDRIVEDataset(root_dir, split='train', size=size)
    val_set = DRIVEDataset(root_dir, split='val', size=size)
    test_set = DRIVEDataset(root_dir, split='test', size=size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader, train_loader, val_loader
