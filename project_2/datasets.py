from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import random 

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

# For the first Two Stream Network attempt
class TwoStreamDataset(Dataset):
    """
    1. Spatial Stream : 10 RGB frames stacked over time -> [3, 10, H, W]
    2. Temporal Stream: Stack of 9 optical flows (u, v) -> [18, H, W]
    """
    def __init__(self, 
                 root_dir='/dtu/datasets1/02516/ucf101_noleakage', 
                 split='train', 
                 spatial_transform=None,
                 temporal_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.is_train = (split == "train")
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        
        self.video_paths = sorted(glob(os.path.join(root_dir, 'videos', split, '*', '*.avi')))
        self.df = pd.read_csv(os.path.join(root_dir, 'metadata', f'{split}.csv'))
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def load_flows(self, flow_dir):
        flow_channels = []
        for i in range(1, self.n_sampled_frames):
            flow_filename = f"flow_{i}_{i+1}.png"
            flow_path = os.path.join(flow_dir, flow_filename)
            flow_img = Image.open(flow_path).convert("RGB")
            
            if self.temporal_transform:
                flow_tensor = self.temporal_transform(flow_img)
            else:
                flow_tensor = T.ToTensor()(flow_img)
            
            # Extract u and v channels (first 2 channels)
            flow_uv = flow_tensor[:2, :, :] 
            flow_channels.append(flow_uv)

        # [18, H, W]
        stacked_flow = torch.cat(flow_channels, dim=0)
        return stacked_flow

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        class_name = os.path.basename(os.path.dirname(video_path))
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # SPATIAL STREAM
        frames_base_dir = os.path.join(self.root_dir, 'frames', self.split, class_name, video_name)
        frame_tensors = []

        for frame_idx in range(1, self.n_sampled_frames + 1):
            frame_path = os.path.join(frames_base_dir, f"frame_{frame_idx}.jpg")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frames_base_dir, f"frame_{frame_idx:05d}.jpg")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frames_base_dir, "frame_1.jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(frames_base_dir, "frame_00001.jpg")

            frame_img = Image.open(frame_path).convert("RGB")
            if self.spatial_transform:
                frame_t = self.spatial_transform(frame_img)
            else:
                frame_t = T.ToTensor()(frame_img)
            
            frame_tensors.append(frame_t)  
        spatial_video = torch.stack(frame_tensors, dim=0).permute(1, 0, 2, 3)

        # TEMPORAL STREAM
        flow_dir = os.path.join(self.root_dir, 'flows_png', self.split, class_name, video_name)
        temporal_flows = self.load_flows(flow_dir)

        return (spatial_video, temporal_flows), label

def GetTwoStreamLoaders(root_dir='/dtu/datasets1/02516/ucf101_noleakage', batch_size=8, num_workers=2):
    # With DataAugm
    train_spatial_transform = T.Compose([
        T.Resize((72, 72)),
        T.RandomCrop((64, 64)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_spatial_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Resize only!
    temporal_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    train_set = TwoStreamDataset(root_dir, split="train", 
                                 spatial_transform=train_spatial_transform, 
                                 temporal_transform=temporal_transform)
    val_set   = TwoStreamDataset(root_dir, split="val", 
                                 spatial_transform=eval_spatial_transform, 
                                 temporal_transform=temporal_transform)
    test_set  = TwoStreamDataset(root_dir, split="test", 
                                 spatial_transform=eval_spatial_transform, 
                                 temporal_transform=temporal_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader, train_loader, val_loader

#                     V2 Two Stream Network 

class SpatialDatasetV2(Dataset):
    def __init__(self, root_dir='/dtu/datasets1/02516/ucf101_noleakage', split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.is_train = (split == 'train')
        self.transform = transform
        self.n_sampled_frames = 10
        self.df = pd.read_csv(os.path.join(root_dir, 'metadata', f'{split}.csv'))

        if self.is_train:
            self.frame_paths = sorted(glob(os.path.join(root_dir, 'frames', split, '*', '*', '*.jpg')))
        else:
            self.video_paths = sorted(glob(os.path.join(root_dir, 'videos', split, '*', '*.avi')))

    def __len__(self):
        return len(self.frame_paths) if self.is_train else len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        if self.is_train:
            frame_path = self.frame_paths[idx]
            video_name = os.path.basename(os.path.dirname(frame_path))
            label = self._get_meta('video_name', video_name)['label'].item()
            
            frame_img = Image.open(frame_path).convert("RGB")
            frame_tensor = self.transform(frame_img) if self.transform else T.ToTensor()(frame_img)
            return frame_tensor, label
        else:
            video_path = self.video_paths[idx]
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            class_name = os.path.basename(os.path.dirname(video_path))
            label = self._get_meta('video_name', video_name)['label'].item()

            frames_dir = os.path.join(self.root_dir, 'frames', self.split, class_name, video_name)
            frames = []
            for i in range(1, self.n_sampled_frames + 1):
                f_path = os.path.join(frames_dir, f"frame_{i}.jpg")
                if not os.path.exists(f_path):
                    f_path = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
                if not os.path.exists(f_path):
                    f_path = os.path.join(frames_dir, "frame_1.jpg")
                
                img = Image.open(f_path).convert("RGB")
                frames.append(self.transform(img) if self.transform else T.ToTensor()(img))
            video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
            return video_tensor, label


class TemporalDatasetV2(Dataset):
    def __init__(self, root_dir='/dtu/datasets1/02516/ucf101_noleakage', split='train', is_train=True):
        self.root_dir = root_dir
        self.split = split
        self.is_train = is_train
        self.n_sampled_frames = 10
        self.video_paths = sorted(glob(os.path.join(root_dir, 'videos', split, '*', '*.avi')))
        self.df = pd.read_csv(os.path.join(root_dir, 'metadata', f'{split}.csv'))
        
        self.base_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        class_name = os.path.basename(os.path.dirname(video_path))
        label = self._get_meta('video_name', video_name)['label'].item()

        flow_dir = os.path.join(self.root_dir, 'flows_png', self.split, class_name, video_name)
        flow_channels = []

        for i in range(1, self.n_sampled_frames):
            flow_file = os.path.join(flow_dir, f"flow_{i}_{i+1}.png")
            flow_img = Image.open(flow_file).convert("RGB")
            flow_tensor = self.base_transform(flow_img)
            flow_uv = (flow_tensor[:2, :, :] - 0.5) * 2.0
            flow_channels.append(flow_uv)
        stacked_flow = torch.cat(flow_channels, dim=0)
        if self.is_train and random.random() < 0.5:
            stacked_flow = torch.flip(stacked_flow, dims=[2])
            stacked_flow[0::2, :, :] = -stacked_flow[0::2, :, :]

        return stacked_flow, label



def GetSpatialV2Loaders(root_dir='/dtu/datasets1/02516/ucf101_noleakage', batch_size=32, num_workers=2):
    train_transform = T.Compose([
        T.RandomResizedCrop(64, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.2), value='random')
    ])
    eval_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = SpatialDatasetV2(root_dir=root_dir, split='train', transform=train_transform)
    val_set   = SpatialDatasetV2(root_dir=root_dir, split='val',   transform=eval_transform)
    test_set  = SpatialDatasetV2(root_dir=root_dir, split='test',  transform=eval_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=8,          shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=8,          shuffle=False, num_workers=num_workers)

    return test_loader, train_loader, val_loader


def GetTemporalV2Loaders(root_dir='/dtu/datasets1/02516/ucf101_noleakage', batch_size=8, num_workers=2):
    train_set = TemporalDatasetV2(root_dir=root_dir, split='train', is_train=True)
    val_set   = TemporalDatasetV2(root_dir=root_dir, split='val',   is_train=False)
    test_set  = TemporalDatasetV2(root_dir=root_dir, split='test',  is_train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader, train_loader, val_loader