import torch
from torchvision.ops import roi_align
from torch.utils.data import DataLoader

from datasets import PotholesDataset
from proposals import PotholeImage, BACKGROUND, POTHOLE

def GetIndexFromId(dataset: PotholesDataset, id: str):
    for i in range(len(dataset)):
        imageId = dataset.paths[i].stem
        if imageId == id:
            return i
    return -1 

def MakeSplit(image: PotholeImage, fracBackground: float, maxProposals: int):
    backgroundIndeces = torch.where(image.proposalLabels == BACKGROUND)[0]
    potholeIndeces = torch.where(image.proposalLabels == POTHOLE)[0]
    
    numBackground = len(backgroundIndeces)
    numPothole = len(potholeIndeces)

    if numPothole == 0 and numBackground > 0:
        numBackground = 1
        numPothole = 0
    else:
        shouldBeBackground = round(numPothole / (1 - fracBackground)) - numPothole
        if len(backgroundIndeces) > shouldBeBackground:
            numBackground = shouldBeBackground
        elif len(backgroundIndeces) == shouldBeBackground:
            pass
        else:
            numPothole = round(numBackground * (1 - fracBackground) / fracBackground)

    newTotal = numBackground + numPothole
    if newTotal > maxProposals:
        fracToRemove = maxProposals / newTotal
        numBackground = round(numBackground * fracToRemove)
        numPothole = round(numPothole * fracToRemove)

    backgroundIndeces = backgroundIndeces[torch.randperm(len(backgroundIndeces))[:numBackground]]
    potholeIndeces = potholeIndeces[torch.randperm(len(potholeIndeces))[:numPothole]]

    allIndeces = torch.cat([backgroundIndeces, potholeIndeces])
    image.proposalLabels = image.proposalLabels[allIndeces]
    image.proposals = image.proposals[allIndeces]


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: PotholesDataset, proposals: dict, ouputImageSize = (128, 128), split='train', maxProposals=1000, fracBackground=0.75):
        self.dataset = dataset
        self.split = split
        self.outputImageSize = ouputImageSize
        self.proposals: list[PotholeImage] = []
        for key, value in proposals.items():
            if split == value.split:
                idx = GetIndexFromId(self.dataset, key)
                if idx == -1:
                    continue
                value.index = idx
                if split == 'train':
                    MakeSplit(value, fracBackground, maxProposals)
                else:
                    value.proposals = value.proposals[:maxProposals] # Otherwise eats ram too much, doresnt bias the splits though
                    value.proposalLabels = value.proposalLabels[:maxProposals]
                self.proposals.append(value)

    def __len__(self):
        return len(self.proposals)

    def __getitem__(self, idx: int):
        proposal = self.proposals[idx]
        datasetIndex = proposal.index

        image, target = self.dataset[datasetIndex]
        boxes = proposal.proposals
        labels = proposal.proposalLabels

        proposalImages = roi_align(image.unsqueeze(0), [boxes],
                                    output_size=self.outputImageSize,
                                    aligned=True)
        
        return proposalImages, labels

def CollateProposals(batch): # Essentially flattens since torch doesnt like different num proposals per image. Thisa makes it all one long list instead of per image indexing
    crops, labels = zip(*batch)
    return torch.cat(crops), torch.cat(labels)

def GetPotholeLoaders(trainPotholeDataset: PotholesDataset, valPotholeDataset: PotholesDataset, testPotholeDataset: PotholesDataset, proposals: dict, batch_size, num_workers, max_proposals, outpute_image_size=(128, 128)):
    train_set = BoxDataset(trainPotholeDataset, proposals, split='train', maxProposals=max_proposals, ouputImageSize=outpute_image_size)
    val_set = BoxDataset(valPotholeDataset, proposals, split='val', maxProposals=max_proposals, ouputImageSize=outpute_image_size)
    test_set = BoxDataset(testPotholeDataset, proposals, split='test', maxProposals=max_proposals, ouputImageSize=outpute_image_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=CollateProposals)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=CollateProposals)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=CollateProposals)
    
    return test_loader, train_loader, val_loader

def GetPotholeLoadersMinusTest(trainPotholeDataset: PotholesDataset, valPotholeDataset: PotholesDataset, proposals: dict, batch_size, num_workers, max_proposals, outpute_image_size=(128, 128)):
    train_set = BoxDataset(trainPotholeDataset, proposals, split='train', maxProposals=max_proposals, ouputImageSize=outpute_image_size)
    val_set = BoxDataset(valPotholeDataset, proposals, split='val', maxProposals=max_proposals, ouputImageSize=outpute_image_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=CollateProposals)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=CollateProposals)
    
    return train_loader, val_loader