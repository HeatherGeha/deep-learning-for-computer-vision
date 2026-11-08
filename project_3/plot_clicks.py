"""Report figure: simulated clicks on PH2 training images, one strategy per column.

Shows three images with a small, medium and large lesion. The white line in the
click columns is the true lesion edge.

Usage: python plot_clicks.py [--n_pos 5] [--n_neg 5] [--out ph2_clicks.pdf]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from datasets_PH2 import CLICK_STRATEGIES, PH2_ROOT, PH2ClickDataset, PH2Dataset

POSITIVE_COLOR = '#1baf7a'
NEGATIVE_COLOR = '#e0364b'
INK = '#0b0b0b'
LESION_QUANTILES = (0.15, 0.5, 0.85)
MARKERS = {
    'positive': dict(marker='o', color=POSITIVE_COLOR, label='Positive click (lesion)'),
    'negative': dict(marker='s', color=NEGATIVE_COLOR, label='Negative click (background)'),
}


def draw_clicks(ax, clicks, value, marker, color, label):
    ys, xs = (clicks[0] == value).nonzero(as_tuple=True)
    ax.scatter(xs, ys, marker=marker, s=34, c=color, edgecolors='white', linewidths=0.8, label=label,
               clip_on=False, zorder=3)


def plot_clicks(root_dir, n_pos, n_neg, out):
    datasets = {s: PH2ClickDataset(root_dir, n_pos, n_neg, strategy=s, flip=False) for s in CLICK_STRATEGIES}
    first = datasets[CLICK_STRATEGIES[0]]

    def full_sample(i):
        return PH2Dataset.__getitem__(first, i)

    by_lesion_size = sorted(range(len(first)), key=lambda i: full_sample(i)[1].mean().item())
    rows = [by_lesion_size[int(q * (len(first) - 1))] for q in LESION_QUANTILES]

    titles = ['Full mask'] + [f'{s.capitalize()} clicks' for s in CLICK_STRATEGIES]
    fig, axes = plt.subplots(len(rows), len(titles), figsize=(7, 4.9), squeeze=False)

    for r, i in enumerate(rows):
        image, mask = full_sample(i)
        mask_np = mask[0].numpy()

        axes[r, 0].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f'{first.ids[i]}\n{100 * mask_np.mean():.0f}% lesion', color=INK, fontsize=9)

        for c, strategy in enumerate(CLICK_STRATEGIES, start=1):
            ax = axes[r, c]
            clicks = datasets[strategy][i][1]
            ax.imshow(image.permute(1, 2, 0).numpy())
            ax.contour(mask_np, levels=[0.5], colors='white', linewidths=0.8)
            draw_clicks(ax, clicks, 1, **MARKERS['positive'])
            draw_clicks(ax, clicks, 0, **MARKERS['negative'])

    for c, title in enumerate(titles):
        axes[0, c].set_title(title, color=INK, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [Line2D([], [], linestyle='none', markersize=7, markeredgecolor='white',
                      marker=m['marker'], markerfacecolor=m['color'], label=m['label'])
               for m in MARKERS.values()]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.6)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}: {n_pos} positive and {n_neg} negative clicks, images {[first.ids[i] for i in rows]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default=PH2_ROOT)
    parser.add_argument('--n_pos', type=int, default=5)
    parser.add_argument('--n_neg', type=int, default=5)
    parser.add_argument('--out', default='ph2_clicks.pdf')
    args = parser.parse_args()
    plot_clicks(args.root_dir, args.n_pos, args.n_neg, args.out)
