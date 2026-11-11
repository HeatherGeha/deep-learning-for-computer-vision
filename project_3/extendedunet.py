import torch
import torch.nn as nn
from unet import UNet
from losses import DiceFocalLoss

class ExtendedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()

    def forward(self, x):
        return self.unet(x)

def GetNetworkOptimizerCriterionAndScheduler(device, num_epochs):
    """
    Instantiates U-Net with Adam optimizer, DiceFocalLoss, and CosineAnnealingLR.
    """
    model = ExtendedUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DiceFocalLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    return model, optimizer, criterion, scheduler