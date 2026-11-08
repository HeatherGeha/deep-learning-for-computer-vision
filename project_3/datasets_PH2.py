"""PH2 skin lesion dataset.

Layout: <root>/IMD002/IMD002_Dermoscopic_Image/IMD002.bmp  (image)
        <root>/IMD002/IMD002_lesion/IMD002_lesion.bmp      (lesion mask)
The IMD*_roi folders are not used. PH2 has no pre-assigned splits.

Run `python datasets_PH2.py` on the HPC to check the data and print its stats.
"""
from glob import glob
import os

from PIL import Image
import torch
from torch.utils.data import DataLoader

from seg_common import split_ids, transform_sample

PH2_ROOT = '/dtu/datasets1/02516/PH2_Dataset_images'
# About the 3:4 shape of the original images, and divisible by 32 for the U-Net
PH2_SIZE = (288, 384)  # (height, width)
PH2_N_VAL = 30
PH2_N_TEST = 30


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
