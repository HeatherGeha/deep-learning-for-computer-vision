
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

    def collect_probs(self, spatial_loader, temporal_loader, device=None):
        """
        Runs both streams over a split and returns (spatial_probs, temporal_probs, targets).
        The two loaders must iterate over the same videos in the same order.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_spatial_probs = []
        all_temporal_probs = []
        all_targets = []

        with torch.no_grad():
            for (s_data, target), (t_data, t_target) in zip(spatial_loader, temporal_loader, strict=True):
                if not torch.equal(target, t_target):
                    raise RuntimeError("Spatial and temporal loaders are not aligned (labels differ).")
                s_data, t_data = s_data.to(device), t_data.to(device)

                s_probs = torch.exp(self.spatial_stream(s_data))
                t_probs = torch.softmax(self.temporal_stream(t_data), dim=1)

                all_spatial_probs.append(s_probs.cpu())
                all_temporal_probs.append(t_probs.cpu())
                all_targets.append(target.cpu())

        return torch.cat(all_spatial_probs), torch.cat(all_temporal_probs), torch.cat(all_targets)

    def evaluate_fusion_sweep(self, val_spatial_loader, val_temporal_loader,
                              test_spatial_loader, test_temporal_loader, device=None, step=0.05):
        """
        Picks the fusion weight alpha with a grid sweep on the VALIDATION set,
        then evaluates once on the test set with that alpha.
        """
        def accuracy(probs, targets):
            return (torch.argmax(probs, dim=1) == targets).float().mean().item() * 100.0

        val_s, val_t, val_y = self.collect_probs(val_spatial_loader, val_temporal_loader, device)

        print("=" * 64)
        print("          LATE FUSION SCORE SWEEP (VALIDATION SET)              ")
        print("=" * 64)
        print(f"{'Spatial Weight (alpha)':<22} | {'Flow Weight (1-alpha)':<22} | {'Val Accuracy':<10}")
        print("-" * 64)

        best_acc = -1.0
        best_alpha = 0.0

        steps = int(round(1.0 / step)) + 1
        for i in range(steps):
            alpha = round(i * step, 4)
            acc = accuracy(alpha * val_s + (1.0 - alpha) * val_t, val_y)

            highlight = " <-- BEST" if acc > best_acc else ""
            print(f"{alpha:<22.2f} | {(1.0 - alpha):<22.2f} | {acc:.2f}%{highlight}")

            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha

        print("=" * 64)
        print(f"Optimal Alpha Found (val) : {best_alpha:.2f} (Spatial: {best_alpha*100:.0f}%, Flow: {(1-best_alpha)*100:.0f}%)")
        print(f"Fused Val Accuracy        : {best_acc:.2f}%\n")

        self.fusion_weight = best_alpha

        test_s, test_t, test_y = self.collect_probs(test_spatial_loader, test_temporal_loader, device)
        test_acc = accuracy(best_alpha * test_s + (1.0 - best_alpha) * test_t, test_y)

        print(f"Test Spatial Stream Alone : {accuracy(test_s, test_y):.2f}%")
        print(f"Test Temporal Stream Alone: {accuracy(test_t, test_y):.2f}%")
        print(f"Test Fused (alpha={best_alpha:.2f})   : {test_acc:.2f}%")

        return best_alpha, test_acc

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
