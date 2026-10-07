import torch
import torch.nn as nn

# 2D CNN + early fusion
# The 10 frames are combined at the input: [B, 3, 10, H, W] -> [B, 30, H, W]
# and then processed by a normal 2D CNN.
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
            # 10 frames x 3 colour channels = 30 input channels
            nn.Conv2d(30, 32, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )

        # 64x64 -> 32x32 -> 16x16 -> 8x8
        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # x: [batch, 3, 10, 64, 64] -> [batch, 30, 64, 64]
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)

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
