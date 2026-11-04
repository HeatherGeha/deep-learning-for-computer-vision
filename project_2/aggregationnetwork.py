import torch
import torch.nn as nn

# 2D CNN + per-frame aggregation
# The network is trained on single frames (FrameImageDataset).
# At test time it is applied to all 10 frames of a video and the softmax
# predictions are averaged over the frames.
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
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

            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )

        # Input images are 64x64, so after 3 pooling layers: 64 -> 32 -> 16 -> 8
        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # x: [batch, 3, 64, 64] (single frames)
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

def GetNetworkOptimizerCriterionAndScheduler(device: torch.device, num_epochs: int):
    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.85, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    return model, optimizer, criterion, scheduler

def EvaluateAggregated(device: torch.device, model: torch.nn.Module, video_loader: torch.utils.data.DataLoader) -> float:
    """Video-level accuracy: average the per-frame softmax over the frames of each video."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for videos, labels in video_loader:
            videos, labels = videos.to(device), labels.to(device)
            B, C, T, H, W = videos.shape

            # [B, 3, T, H, W] -> [B*T, 3, H, W]
            frames = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            probs = torch.softmax(model(frames), dim=1)

            # [B*T, 10] -> [B, T, 10] -> average over frames
            probs = probs.reshape(B, T, -1).mean(dim=1)
            correct += (probs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total
