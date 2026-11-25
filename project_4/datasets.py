from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split # we are instructed to use this to make up for missing splits.json file


POTHOLES_ROOT = '/dtu/datasets1/02516/potholes'
RANDOM_STATE = 42 #must not be changed, this was in instructions


class PotholesDataset(Dataset):

    def __init__(self, root_dir=POTHOLES_ROOT, split='train', size=None):
        self.root = Path(root_dir)
        self.size = size  # (height, width) for resizing 
        # NOTE: resizing is an efficiency thing - if runs are slow, do it. If they are not, you needn't bother

        paths = sorted((self.root / 'images').glob('*.png'))

        # train test split given to us in instructions - don't change
        train, test = train_test_split(paths, test_size=0.1, random_state=RANDOM_STATE)
        # 10% of training reserved for validation
        train, val = train_test_split(train, test_size=0.1, random_state=RANDOM_STATE)

        self.paths = {'train': train,'val': val,'test': test}[split]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # load image and record its original dimensons
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')
        width, height = image.size

        #locate and parse corresponding annotations file
        #following link1, link2 instructions from problem formulation
        annotation = ET.parse(self.root / 'annotations' / f'{path.stem}.xml')

        #extract all ground-truth bounding boxes from the image
        boxes = []

        for box in annotation.findall('.//bndbox'):
            boxes.append([
                float(box.findtext('xmin')),
                float(box.findtext('ymin')),
                float(box.findtext('xmax')),
                float(box.findtext('ymax'))
            ])

        #convert bounding boxes to a tensor
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        #resize image if target size input
        if self.size is not None:
            new_h, new_w = self.size
            image = TF.resize(image, self.size)
            boxes[:, [0, 2]] *= new_w / width
            boxes[:, [1, 3]] *= new_h / height

        image = TF.to_tensor(image)

        # target dictionary: all ground-truth objects belong to class 1 which is 'pothole'
        target = {'boxes': boxes,'labels': torch.ones(len(boxes), dtype=torch.int64)}

        return image, target