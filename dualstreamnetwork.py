import torch
import torch.nn as nn

#
#
# This whole code was changed, there was supposed to be a structural backbone model and then both the spatial and temporal stream used that structure, but I tried to change it but then decided to do V2 so I could really do both convnet seperatlly. 
# So this model was not really the one used to produce the earlier results, its more of an artifact now.
#

class SpatialStream(nn.Module):
    """
    Spatial ConvNet with per-frame aggregation capability.
    - Training: Accepts single frames [B, 3, H, W] or video stacks [B, 3, T, H, W]
    - Evaluation: Aggregates across T frames via softmax averaging
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super(SpatialStream, self).__init__()
        
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)   # 16x16 -> 8x8
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            features = self.convolutional(frames)
            features = features.view(features.size(0), -1)
            logits = self.fully_connected(features)
            probs = torch.softmax(logits, dim=1).reshape(B, T, -1).mean(dim=1)
            return torch.log(probs + 1e-8)

        features = self.convolutional(x)
        features = features.view(features.size(0), -1)
        return self.fully_connected(features)


class TemporalStream(nn.Module):
    """
    Temporal ConvNet: processes 9 stacked optical flows (18 channels).

    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super(TemporalStream, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)   # 16x16 -> 8x8
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.convolutional(x)
        features = features.view(features.size(0), -1)
        return self.fully_connected(features)


class DualStreamNetwork(nn.Module):
    """
    Late Fusion of Spatial Stream (with Aggregation) + Temporal Stream (Optical Flow)
    """
    def __init__(self, fusion_weight: float = 0.2):
        super(DualStreamNetwork, self).__init__()
        self.fusion_weight = fusion_weight
        self.spatial_stream = SpatialStream(num_classes=10, dropout=0.6)
        self.temporal_stream = TemporalStream(num_classes=10, dropout=0.2)

    def forward(self, inputs):
        spatial_x, temporal_x = inputs[0], inputs[1]
        
        spatial_logits = self.spatial_stream(spatial_x)
        temporal_logits = self.temporal_stream(temporal_x)

        fused_logits = (self.fusion_weight * spatial_logits) + ((1.0 - self.fusion_weight) * temporal_logits)
        return fused_logits


def GetNetworkOptimizerCriterionAndScheduler(device: torch.device, num_epochs: int):
    """
    Configures asymmetric learning rates and regularizations per stream.
    """
    model = DualStreamNetwork(fusion_weight=0.2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.Adam([
        {
            'params': model.spatial_stream.parameters(),
            'lr': 0.001,
            'weight_decay': 1e-3
        },
        {
            'params': model.temporal_stream.parameters(),
            'lr': 0.001,
            'weight_decay': 1e-4
        }
    ], betas=(0.85, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        eta_min=1e-5
    )

    return model, optimizer, criterion, schedulerbjob