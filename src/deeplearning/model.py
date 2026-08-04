"""
model.py

A simple 3D CNN for brain MRI.
"""

import torch
import torch.nn as nn


class BrainAgeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1)

        )

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64,32),

            nn.ReLU(),

            nn.Linear(32,1)

        )

    def forward(self,x):

        x = self.features(x)

        x = self.regressor(x)

        return x


if __name__ == "__main__":

    model = BrainAgeCNN()
s
    x = torch.randn(2,1,212,210,128)

    y = model(x)

    print(model)

    print("Input shape :",x.shape)

    print("Output shape:",y.shape)