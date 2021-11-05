"""
Convolutional Autoencoder

1) Goal: Reconstruct any CIFAR-10 image
- Cost function: MSE or binary cross-entropy between input and reconstructed image
- Activation for conv layers: ReLU
- Architecture:
    Input: 32x32x3 (colored image)
    Encoder:
        Layer 1: Filter: 3x3 and 8 channels -> Max pooling 2x2
        Layer 2: Filter: 3x3 + Channels: 12 -> Max pooling 2x2
    Decoder:
        Layer 3: Filter: 3x3 and 16 channels -> Upsampling 2x2
        Layer 4: Filter: 3x3 and 12 channels -> Upsampling 2x2
        Layer 5: Filter: 3x3 and 3 channels
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import load_CIFAR10, plot_img_dataloader

train_loader, test_loader, valid_loader, classes = load_CIFAR10()

# https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/

"""class CAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),  # Out = 48x48x64 48=48-5+2*2+1 where K=64
            nn.ELU(),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Out = 48x48x128 and K=32
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.4)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Out = 48x48x256 and K=32
            nn.ELU(),
            nn.BatchNorm2d(256)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Out = 48x48x256 and K=32
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.4)
        )

        self.flat = nn.Sequential(
            nn.Flatten()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(36864, 128),
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.6)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, NUM_LABELS),  # 7 classes output
            # nn.LogSoftmax(dim=1) # IMPORTANT: No Softmax must be applied in the last layer if we use the Cross-Entropy Loss
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv3(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flat(x)
        # x = x.view(-1, 2 * 2 * 128)
        #x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # y = self.fc2(x)
        y = checkpoint(self.fc2, x)
        return y"""