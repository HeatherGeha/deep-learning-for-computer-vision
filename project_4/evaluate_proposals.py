import torch
from torchvision.ops import box_iou

def EvaluateProposals(results, proposal_counts, iou_threshold=0.5):
    """Evaluate proposal recall for different numbers of proposals."""

    total_objects = 0
    total_proposals = 0
    num_images = 0
    detected = {k: 0 for k in proposal_counts}

    for sample in results.values():

        # only evaluate on the training set
        if sample['split'] != 'train':
            continue

        proposals = sample['proposals']
        boxes = sample['boxes']

        total_objects += len(boxes)
        total_proposals += len(proposals)
        num_images += 1

        # Compute pairwise IoU between proposals and ground truth
        # Shape: [number of proposals, number of ground-truth boxes]
        iou = box_iou(proposals, boxes)

        for k in proposal_counts:

            # Consider only the first k proposals
            selected = iou[:k]

            # Find the best matching proposal for each ground-truth box
            best_iou = selected.max(dim=0).values

            # Count successfully covered ground-truth objects
            detected[k] += (best_iou >= iou_threshold).sum().item()

    # Calculate recall for each proposal count
    recall = {
        k: detected[k] / total_objects
        for k in proposal_counts
    }

    print('\n--- Selective Search: Training Set Evaluation ---')
    print(f'Total training images evaluated: {num_images}')
    print(f'Total ground-truth potholes: {total_objects}')
    print(f'Mean proposals per image: {total_proposals / num_images:.1f}')
    print(f'IoU threshold: {iou_threshold}')

    print('\nAggregate recall across all training images:')
    print(f'{"Max proposals/image":>20} | {"Potholes covered":>20} | {"Recall":>8}')
    print('-' * 55)

    for k in proposal_counts:
        print(f'{k:>20} | {detected[k]:>8}/{total_objects:<11} | {recall[k]:>7.2%}')

    return recall
