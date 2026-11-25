# PART 1
- Familiarise yourself with the data and visualize some examples with the
ground-truth bounding boxes
- Extract object proposals for all the images of the dataset (e.g. Selecting
Search, Edge Boxes, etc). Note that you may have to resize the images
before you run SS for better efficiency.
- Evaluate the extracted proposals on the training set of the dataset and
determine the number of required proposals
- Prepare the proposals for the training of the object detector. This requires
assigning a label (i.e., class or background label) to each proposal.

## Part 1 status: done
| Task | File | How to run |
|---|---|---|
| 1. Visualise GT boxes | `visualize.py` | `python visualize.py` (HPC) → `figures/gt_examples.png` |
| 2. Extract proposals | `proposals.py`, `main.py` | already generated: `ss_fast_original.pt`, `ss_quality_original.pt` |
| 3. Evaluate proposals | `evaluate_proposals.py`, `main.py` | `python main.py --mode quality` |
| 4. Label proposals | `label_proposals.py` | `python label_proposals.py` → `labeled_proposals.pt` (not in git, takes seconds) |

**Chosen proposals (Heather):** SS quality mode, K = 5000 per image. Train recall@0.5 = 92.9%.

**Labelling:** IoU ≥ 0.5 → pothole (1), IoU < 0.3 → background (0), 0.3–0.5 dropped, plus the GT boxes as extra positives.
Train has 50,911 pothole and 1,246,328 background proposals (**24.5:1**); val has 6,320 and 128,029 (20:1).
Part 2 needs to handle this imbalance.
`labeled_proposals.pt` format: `{image_id: {'boxes': [N,4] xyxy, 'labels': [N], 'ious': [N], 'split'}}` (train/val only).

# PART 2
- Build a convolutional neural network to classify object proposals (N+1
classes)
- Build a dataloader for the object detection task. Think about the class
imbalance issue of the background proposals
- Finetune the network on the training set
- Evaluate the classification accuracy of the network on the validation set.
Note that this is different from the evaluation of the object detection tas

# PART 3
- Apply the CNN that you trained on the test images
- Implement and apply NMS to discard overlapping boxes
- Evaluate the object detection output using the Average Precision (AP)
metric