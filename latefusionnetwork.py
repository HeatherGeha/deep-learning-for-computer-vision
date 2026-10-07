import torch
import torch.nn as nn

# 2D CNN + late fusion
# The same 2D CNN (shared weights) is applied independently to all 10 frames.
# The high-level per-frame features are combined before the classifier.
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
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

        # One 128-dim feature vector per frame
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # x: [batch, 3, 10, 64, 64]
        B, C, T, H, W = x.shape

        # Put frames into the batch dimension: [B*10, 3, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        x = self.convolutional(x)

        # [B*10, 128, 8, 8] -> [B, 10, 128]
        x = self.spatial_pool(x).reshape(B, T, 128)

        # Late fusion: combine the frame features
        x = x.mean(dim=1)

        x = self.classifier(x)
        return x

def GetNetworkOptimizerCriterionAndScheduler(device: torch.device, num_epochs: int):
    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.85, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    return model, optimizer, criterion, scheduler
