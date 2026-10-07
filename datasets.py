from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='/work3/ppar/data/ucf101',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(os.path.join(root_dir, 'frames', split, '*', '*', '*.jpg')))
        self.df = pd.read_csv(os.path.join(root_dir, 'metadata', f'{split}.csv'))
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = os.path.basename(os.path.dirname(frame_path))
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir = '/work3/ppar/data/ucf101', 
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.root_dir = root_dir
        self.video_paths = sorted(glob(os.path.join(root_dir, 'videos', split, '*', '*.avi')))
        self.df = pd.read_csv(os.path.join(root_dir, 'metadata', f'{split}.csv'))
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        class_name = os.path.basename(os.path.dirname(video_path))
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = os.path.join(self.root_dir, 'frames', self.split, class_name, video_name)
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames

from torch.utils.data import DataLoader
def GetFrameLoaders(root_dir: str): 
    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_test_dataset = FrameImageDataset(root_dir=root_dir, split='test', transform=transform)
    frameimage_test_loader = DataLoader(frameimage_test_dataset,  batch_size=8, shuffle=False)

    frameimage_train_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
    frameimage_train_loader = DataLoader(frameimage_train_dataset,  batch_size=8, shuffle=True)

    frameimage_val_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    frameimage_val_loader = DataLoader(frameimage_val_dataset,  batch_size=8, shuffle=False)

    return frameimage_test_loader, frameimage_train_loader, frameimage_val_loader
            
def GetVideoStackLoaders(root_dir: str): 
    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    framevideostack_test_dataset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames = True)
    framevideostack_test_loader = DataLoader(framevideostack_test_dataset,  batch_size=8, shuffle=False)

    framevideostack_train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames = True)
    framevideostack_train_loader = DataLoader(framevideostack_train_dataset,  batch_size=8, shuffle=True)

    framevideostack_val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideostack_val_loader = DataLoader(framevideostack_val_dataset,  batch_size=8, shuffle=False)

    return framevideostack_test_loader, framevideostack_train_loader, framevideostack_val_loader

def GetVideoListLoaders(root_dir: str):
    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    framevideolist_test_dataset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames = False)
    framevideolist_test_loader = DataLoader(framevideolist_test_dataset,  batch_size=8, shuffle=False)

    framevideolist_train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames = False)
    framevideolist_train_loader = DataLoader(framevideolist_train_dataset,  batch_size=8, shuffle=True)

    framevideolist_val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)
    framevideolist_val_loader = DataLoader(framevideolist_val_dataset,  batch_size=8, shuffle=False)

    return framevideolist_test_loader, framevideolist_train_loader, framevideolist_val_loader 