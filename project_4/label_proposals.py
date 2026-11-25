"""Part 1, task 4: give every proposal a label (1 = pothole, 0 = background).

Only needs the proposal cache, not the images, so it runs in seconds anywhere:
    python label_proposals.py                      # quality proposals, K=5000
    python label_proposals.py --k 1000 --pos 0.5 --neg 0.3

Labelling rule (R-CNN style):
    best IoU with any GT box >= POS_IOU  -> 1 (pothole)
    best IoU                 <  NEG_IOU  -> 0 (background)
    anything in between                  -> dropped (too ambiguous to train on)
The GT boxes themselves are added as extra positives (train/val only).

Output (labeled_proposals.pt): {image_id: {'boxes': [N, 4] float (xmin, ymin, xmax, ymax),
                                           'labels': [N] int64, 'ious': [N] float,
                                           'split': 'train' | 'val'}}
Test images are not labelled: in Part 3 they are run through the detector using the raw
proposals from the cache and scored against the GT boxes with AP.
"""
import argparse

import torch
from torchvision.ops import box_iou

PROPOSALS_PATH = 'ss_quality_original.pt'
OUTPUT_PATH = 'labeled_proposals.pt'
NUM_PROPOSALS = 5000   # Heather's choice for quality mode (see main.py)
POS_IOU = 0.5
NEG_IOU = 0.3
MIN_GT_SIDE = 2        # a few GT annotations are degenerate (width 0); don't use them as crops


def LabelImage(proposals, gt_boxes, k=NUM_PROPOSALS, pos_iou=POS_IOU, neg_iou=NEG_IOU,
               add_gt=True):
    """Label the top-k proposals of one image. Returns (boxes, labels, ious)."""
    proposals = proposals[:k]

    if len(gt_boxes) == 0:
        return proposals, torch.zeros(len(proposals), dtype=torch.int64), torch.zeros(len(proposals))

    best_iou = box_iou(proposals, gt_boxes).max(dim=1).values
    keep = (best_iou >= pos_iou) | (best_iou < neg_iou)
    boxes, ious = proposals[keep], best_iou[keep]
    labels = (ious >= pos_iou).long()

    if add_gt:
        w = gt_boxes[:, 2] - gt_boxes[:, 0]
        h = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt = gt_boxes[(w >= MIN_GT_SIDE) & (h >= MIN_GT_SIDE)]
        boxes = torch.cat([boxes, gt])
        labels = torch.cat([labels, torch.ones(len(gt), dtype=torch.int64)])
        ious = torch.cat([ious, torch.ones(len(gt))])

    return boxes, labels, ious


def LabelAllProposals(results, k=NUM_PROPOSALS, pos_iou=POS_IOU, neg_iou=NEG_IOU,
                      splits=('train', 'val')):
    labeled = {}
    for image_id, sample in results.items():
        if sample['split'] not in splits:
            continue
        boxes, labels, ious = LabelImage(sample['proposals'], sample['boxes'], k, pos_iou, neg_iou)
        labeled[image_id] = {'boxes': boxes, 'labels': labels, 'ious': ious,
                             'split': sample['split']}
    return labeled


def PrintLabelStats(labeled):
    print(f'\n{"split":>6} | {"images":>6} | {"pothole":>8} | {"background":>10} | {"bg:pothole":>10}')
    print('-' * 55)
    for split in ('train', 'val'):
        per_image = [s['labels'] for s in labeled.values() if s['split'] == split]
        if not per_image:
            continue
        labels = torch.cat(per_image)
        pos, neg = (labels == 1).sum().item(), (labels == 0).sum().item()
        print(f'{split:>6} | {len(per_image):>6} | {pos:>8} | {neg:>10} | {neg / max(pos, 1):>9.1f}x')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposals', default=PROPOSALS_PATH)
    parser.add_argument('--output', default=OUTPUT_PATH)
    parser.add_argument('--k', type=int, default=NUM_PROPOSALS)
    parser.add_argument('--pos', type=float, default=POS_IOU)
    parser.add_argument('--neg', type=float, default=NEG_IOU)
    args = parser.parse_args()

    results = torch.load(args.proposals, weights_only=False)
    labeled = LabelAllProposals(results, args.k, args.pos, args.neg)
    print(f'Proposals: {args.proposals} | K={args.k} | pothole IoU>={args.pos} | background IoU<{args.neg}')
    PrintLabelStats(labeled)

    torch.save(labeled, args.output)
    print(f'\nSaved {args.output}')


if __name__ == '__main__':
    main()
