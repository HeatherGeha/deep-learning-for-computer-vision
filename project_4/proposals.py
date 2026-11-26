import cv2 #using openCV - you might need to update your environment, I did
import numpy as np
import torch
from torchvision.ops import box_iou
import time
from datasets import PotholesDataset

BACKGROUND, POTHOLE = 0, 1

class PotholeImage():
    def __init__(self, proposals: torch.Tensor, proposalLabels: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, imageSize: tuple, split: str, imageId: str):
        self.proposals = proposals
        self.proposalLabels = proposalLabels
        self.boxes = boxes
        self.labels = labels
        self.imageSize = imageSize
        self.split = split
        self.imageId = imageId
        self.index = -1

def SaveProposals(results: dict[str, PotholeImage], path: str):
    torch.save({
        image_id: {
            'proposals': p.proposals,
            'proposal_labels': p.proposalLabels,
            'boxes': p.boxes,
            'labels': p.labels,
            'image_size': p.imageSize,
            'split': p.split,
            'image_id': p.imageId
        }
        for image_id, p in results.items()
    }, path)


def LoadProposals(path: str) -> dict[str, PotholeImage]:
    results = {}
    for image_id, d in torch.load(path).items():
        p = PotholeImage(d['proposals'], d['proposal_labels'], d['boxes'], d['labels'], d['image_size'], d['split'], image_id)
        results[image_id] = p
    return results

def LabelProposals(proposals: torch.Tensor, gt_boxes: torch.Tensor,
                   pos_iou: float = 0.5) -> torch.Tensor:
    labels = torch.full((len(proposals),), BACKGROUND, dtype=torch.long)
    if len(gt_boxes) == 0:
        labels[:] = BACKGROUND
        return labels
    best_iou = box_iou(proposals, gt_boxes).max(dim=1).values

    labels[best_iou >= pos_iou] = POTHOLE
    labels[best_iou < pos_iou] = BACKGROUND
    return labels


def GenerateProposals(image, mode='fast'):
    """Generate candidate bounding boxes for a single image."""

    # convert tensor to numpy array
    image = image.permute(1, 2, 0).numpy()
    # convert pixel values to [0,255]
    image = (image * 255).astype(np.uint8)
    # convert color encoding to match openCVs BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # use Selective Search method for proposals -can be slow 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    # choose from search mode fast/quality 
    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    elif mode == 'quality':
        ss.switchToSelectiveSearchQuality()
    else:
        raise ValueError(f'Unknown search mode: {mode}')

    # generate bounding box proposals
    proposals = ss.process()

    # ground-truth bounding boxes in format xmin,ymin, xmax, ymax
    # convert openCV outputs to match this format
    proposals[:, 2] += proposals[:, 0]
    proposals[:, 3] += proposals[:, 1]

    return torch.tensor(proposals, dtype=torch.float32)

def GenerateAllProposals(mode='fast', size=None):
    results = {}
    start = time.time()

    # generate all proposals using GenerateProposals on each image
    for split in ('train', 'val', 'test'):
        dataset = PotholesDataset(split=split, size=size)

        for idx in range(len(dataset)):
            image_id = dataset.paths[idx].stem
            image, target = dataset[idx]

            proposals = GenerateProposals(image, mode=mode)
            proposalLabels = LabelProposals(proposals, target['boxes'])

            # store proposals dictionary
            results[image_id] = PotholeImage(proposals, proposalLabels, target['boxes'], target['labels'], tuple(image.shape[1:]), split, image_id)
            #progress report
            if (idx + 1) % 50 == 0 or idx + 1 == len(dataset):
                print(f'{split}: {idx + 1}/{len(dataset)}', flush=True)

    print(f'Processed {len(results)} images')
    print(f'Total time: {time.time() - start:.2f} seconds')

    return results
