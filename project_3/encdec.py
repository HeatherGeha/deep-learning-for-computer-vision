import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 512 -> 256
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 256 -> 128
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)  # 32 -> 16
        #self.enc_conv5 = nn.Conv2d(64, 64, 3, padding=1)
        #self.pool5 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2)  # 128 -> 256
        #self.dec_conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(64, 1, 3, padding=1)
        #self.upsample5 = nn.Upsample(scale_factor=2)  # 256 -> 512
        #self.dec_conv5 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))
        e4 = self.pool4(F.relu(self.enc_conv4(e3)))
        #e5 = self.pool5(F.relu(self.enc_conv5(e4)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e4))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = F.relu(self.dec_conv3(self.upsample3(d2)))
        #d4 = F.relu(self.dec_conv4(self.upsample4(d3)))
        #d5 = self.dec_conv5(self.upsample5(d4))  # no activation
        d4 = self.dec_conv4(self.upsample4(d3))  # no activation
        return d4

class DiceLoss(nn.Module):
    def forward(self, y_pred, y_true):
        loss = 1
        loss -= torch.mean(2 * y_pred * y_true + 1) / (torch.mean(y_pred + y_true) + 1)
        return loss
    


def GetNetworkOptimizerCriterionAndScheduler(device: torch.device, num_epochs: int):
    model = Network().to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.85, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    return model, optimizer, criterion, scheduler