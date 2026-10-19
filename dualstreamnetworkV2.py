
import os
import torch
import torch.nn as nn


class SpatialStreamV2(nn.Module):
    """
    Dedicated 2D CNN for appearance recognition.

    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.6):
        super(SpatialStreamV2, self).__init__()

        self.convolutional = nn.Sequential(
            #64x64 -> 32x32
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

            #32x32 -> 16x16
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

            #16x16 -> 8x8
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


class TemporalStreamV2(nn.Module):
    """
    Dedicated 2D CNN for motion dynamics.
    - Input: 18-channel optical flow stack [B, 18, 64, 64]
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super(TemporalStreamV2, self).__init__()

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
            nn.MaxPool2d(2, 2),

            # 32 -> 64 channels
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

            #64 -> 128 channels
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


class DualStreamNetworkV2(nn.Module):
    """
    Decoupled Dual-Stream architecture with late fusion.
    """
    def __init__(self, fusion_weight: float = 0.2, spatial_weights_path: str = None, temporal_weights_path: str = None):
        super(DualStreamNetworkV2, self).__init__()
        self.fusion_weight = fusion_weight  
        self.spatial_stream = SpatialStreamV2(num_classes=10, dropout=0.6)
        self.temporal_stream = TemporalStreamV2(num_classes=10, dropout=0.2)

        if spatial_weights_path and os.path.exists(spatial_weights_path):
            self.spatial_stream.load_state_dict(torch.load(spatial_weights_path, map_location="cpu"))

        if temporal_weights_path and os.path.exists(temporal_weights_path):
            self.temporal_stream.load_state_dict(torch.load(temporal_weights_path, map_location="cpu"))

    def forward(self, inputs):
        spatial_x, temporal_x = inputs[0], inputs[1]

        s_out = self.spatial_stream(spatial_x)
        t_out = self.temporal_stream(temporal_x)

        s_prob = torch.exp(s_out) if spatial_x.dim() == 5 else torch.softmax(s_out, dim=1)
        t_prob = torch.softmax(t_out, dim=1)

        fused_prob = (self.fusion_weight * s_prob) + ((1.0 - self.fusion_weight) * t_prob)
        return torch.log(fused_prob + 1e-8)

    def evaluate_fusion_sweep(self, spatial_loader, temporal_loader, device=None, step=0.05):
        """
        Runs an alpha grid sweep over predictions without having to retrain.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_spatial_probs = []
        all_temporal_probs = []
        all_targets = []

        with torch.no_grad():
            for (s_data, target), (t_data, _) in zip(spatial_loader, temporal_loader):
                s_data, t_data = s_data.to(device), t_data.to(device)

                s_log_probs = self.spatial_stream(s_data)
                s_probs = torch.exp(s_log_probs)

                t_logits = self.temporal_stream(t_data)
                t_probs = torch.softmax(t_logits, dim=1)

                all_spatial_probs.append(s_probs.cpu())
                all_temporal_probs.append(t_probs.cpu())
                all_targets.append(target.cpu())

        all_spatial_probs = torch.cat(all_spatial_probs, dim=0)
        all_temporal_probs = torch.cat(all_temporal_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        min_len = min(all_spatial_probs.shape[0], all_temporal_probs.shape[0], all_targets.shape[0])
        all_spatial_probs = all_spatial_probs[:min_len]
        all_temporal_probs = all_temporal_probs[:min_len]
        all_targets = all_targets[:min_len]

        spatial_acc = (torch.argmax(all_spatial_probs, dim=1) == all_targets).float().mean().item() * 100.0
        temporal_acc = (torch.argmax(all_temporal_probs, dim=1) == all_targets).float().mean().item() * 100.0

        print("=" * 64)
        print("          LATE FUSION SCORE SWEEP (EVALUATION SET)              ")
        print("=" * 64)
        print(f"Isolated Spatial Stream Alone (alpha=1.00) : {spatial_acc:.2f}%")
        print(f"Isolated Temporal Stream Alone (alpha=0.00): {temporal_acc:.2f}%")
        print("-" * 64)
        print(f"{'Spatial Weight (alpha)':<22} | {'Flow Weight (1-alpha)':<22} | {'Accuracy':<10}")
        print("-" * 64)

        best_acc = 0.0
        best_alpha = 0.0

        steps = int(round(1.0 / step)) + 1
        for i in range(steps):
            alpha = round(i * step, 4)
            fused_probs = (alpha * all_spatial_probs) + ((1.0 - alpha) * all_temporal_probs)
            preds = torch.argmax(fused_probs, dim=1)
            acc = (preds == all_targets).float().mean().item() * 100.0

            highlight = " <-- OPTIMAL" if acc > best_acc else ""
            print(f"{alpha:<22.2f} | {(1.0 - alpha):<22.2f} | {acc:.2f}%{highlight}")

            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha

        print("=" * 64)
        print(f"Optimal Alpha Found    : {best_alpha:.2f} (Spatial: {best_alpha*100:.0f}%, Flow: {(1-best_alpha)*100:.0f}%)")
        print(f"Highest Fused Accuracy : {best_acc:.2f}%\n")

        self.fusion_weight = best_alpha
        return best_alpha, best_acc

def GetSpatialOptimizerCriterionAndScheduler(model: nn.Module, num_epochs: int):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.85, 0.999),
        weight_decay=3e-3  
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    return optimizer, criterion, scheduler


def GetTemporalOptimizerCriterionAndScheduler(model: nn.Module, num_epochs: int):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0015,
        betas=(0.85, 0.999),
        weight_decay=1e-4  
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    return optimizer, criterion, scheduler

#filler function, will not get called but needed so main doesnt crash when calling it globally
def GetNetworkOptimizerCriterionAndScheduler(device: torch.device, num_epochs: int):
    model = DualStreamNetworkV2().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    return model, opt, crit, sched