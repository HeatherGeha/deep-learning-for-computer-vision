"""Part 1, task 1: show some train images with their ground-truth boxes.
Needs the images, so run it on the HPC:
    python visualize.py        # saves figures/gt_examples.png
"""
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from datasets import PotholesDataset

OUTPUT_PATH = 'figures/gt_examples.png'


def plot_gt_examples(dataset, n=6, seed=0):
    idxs = random.Random(seed).sample(range(len(dataset)), n)
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 5))

    for ax, idx in zip(axes.flat, idxs):
        image, target = dataset[idx]
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(f'{dataset.paths[idx].stem} ({len(target["boxes"])} potholes)', fontsize=8)
        ax.axis('off')
        for x1, y1, x2, y2 in target['boxes'].tolist():
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor='lime', linewidth=2))

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f'Saved {OUTPUT_PATH}')


if __name__ == '__main__':
    plot_gt_examples(PotholesDataset(split='train'))
