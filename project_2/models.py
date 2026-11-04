import torch
import torch.nn as nn

#2D CNN + per-frame aggregation
#Same 2D CNN applied to all 10 frames, with shared weights.
#Aggregate the frame features/predictions.
#Start with average pooling over the frame dimension, since this matches your notes.
#common version
class Network2D(nn.Module):
    """
    2D CNN that processes one frame at a time.
    Returns a 256-dimensional feature vector.
    """ 
    def __init__(self):
        super(Network2D, self).__init__()

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
        
        # Input images are 64x64, so after 3 pooling layers: 64 -> 32 -> 16 -> 8
        self.features = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.features(x)

        return x

class AggregationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = Network2D()
        self.classifier = nn.Linear(256, 10) #10 classes

    def forward(self, x):
        B, C, T, H, W = x.shape

        # Put frames into the batch dimension
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)

        # Process each frame with the same CNN
        x = self.cnn(x)

        # Put frames back together
        x = x.reshape(B, T, 256)

        # Average the frame features
        x = x.mean(dim=1)

        # Classify
        x = self.classifier(x)

        return x
        
#2D CNN + early fusion
#Combine the 10 frames at the first convolutional layer.
#After that, use a normal 2D CNN.
#This gives temporal information right at the beginning.
class EarlyFusionModel(nn.Module):
    """
    2D CNN with early fusion.
    The 10 frames are combined at the input:
    [B, 3, 10, H, W] -> [B, 30, H, W]
    The 2D CNN then processes the combined frames normally.
    """

    def __init__(self):
        super(EarlyFusionModel, self).__init__()

        self.convolutional = nn.Sequential(
            # 10 RGB frames = 30 input channels
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
            nn.Linear(256, 10) #10 classes
        )

    def forward(self, x):
        # x: [batch, 3, 10, 64, 64]

        # Combine the 10 frames with the colour channels
        # [B, 3, 10, H, W] -> [B, 30, H, W]
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)

        # Normal 2D CNN
        x = self.convolutional(x)

        # Classify
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        return x

#2D CNN + late fusion
#Same 2D CNN applied independently to each frame, shared weights.
#Extract high-level features from each frame.
#Combine the features at the end, before the classifier.
#You can implement the pooling version from your notes rather than the huge concatenated-FC version.
class LateFusionModel(nn.Module):
    """
    2D CNN with late fusion.

    The same 2D CNN is applied independently to all 10 frames.
    The resulting high-level features are averaged before classification.
    """

    def __init__(self):
        super(LateFusionModel, self).__init__()

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

        # Convert the spatial feature maps into a single
        # high-level feature vector for each frame.
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(128, 10) # 10 classes

    def forward(self, x):
        # x: [batch, 3, 10, 64, 64]

        B, C, T, H, W = x.shape

        # Put frames into the batch dimension
        # [B, 3, 10, H, W] -> [B*10, 3, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)

        # Apply the SAME CNN to every frame
        x = self.convolutional(x)

        # Pool each frame's spatial feature map
        # [B*10, 128, 8, 8] -> [B*10, 128, 1, 1]
        x = self.spatial_pool(x)

        # [B*10, 128, 1, 1] -> [B, 10, 128]
        x = x.reshape(B, T, 128)

        # Late fusion:
        # combine the high-level features from the 10 frames
        x = x.mean(dim=1)

        # Classify the whole video
        x = self.classifier(x)

        return x

#3D CNN
#Input: [batch, channels, frames, height, width].
#Use Conv3d instead of Conv2d.
#Temporal and spatial information are processed throughout the network.
class CNN3D(nn.Module):
    """
    3D CNN for video classification.

    Input:
        [batch, channels, frames, height, width]

    Temporal and spatial information are processed
    throughout the network using Conv3d.
    """

    def __init__(self):
        super(CNN3D, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d(2, 2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d(2, 2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool3d(2, 2)
        )

        # Instead of calculating the exact output size,
        # pool everything down to one value per channel.
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Linear(128, 10) #10 classes

    def forward(self, x):
        # x: [batch, channels, frames, height, width]
        # e.g. [B, 3, 10, 64, 64]

        # 3D CNN processes spatial AND temporal dimensions
        x = self.convolutional(x)

        # [B, 128, T, H, W] -> [B, 128, 1, 1, 1]
        x = self.pool(x)

        # [B, 128, 1, 1, 1] -> [B, 128]
        x = x.view(x.size(0), -1)

        # Classify the video
        x = self.classifier(x)

        return x