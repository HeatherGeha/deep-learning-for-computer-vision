import cv2 #using openCV - you might need to update your environment, I did
import numpy as np
import torch
import time
from datasets import PotholesDataset


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

            # store proposals dictionary
            results[image_id] = {
                'proposals': proposals,
                'boxes': target['boxes'],
                'labels': target['labels'],
                'image_size': tuple(image.shape[1:]),
                'split': split
            }

            #progress report
            if (idx + 1) % 50 == 0 or idx + 1 == len(dataset):
                print(f'{split}: {idx + 1}/{len(dataset)}', flush=True)

    print(f'Processed {len(results)} images')
    print(f'Total time: {time.time() - start:.2f} seconds')

    return results
