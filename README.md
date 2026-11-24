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