import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
            # 1 Convolution layer: 3 RGB channels to 32 feature channels + BatchNorm
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # MaxPooling: 128 -> 64
            nn.MaxPool2d(2, 2),

            # 1 Convolution layer: 32 to 64 feature channels + BatchNorm
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # MaxPooling: 64 -> 32
            nn.MaxPool2d(2, 2),

            # 1 Convolution layer: 64 to 128 feature channels + BatchNorm
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # MaxPooling: 32 -> 16
            nn.MaxPool2d(2, 2)
        )

        # Fully connected network
        self.fully_connected = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
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