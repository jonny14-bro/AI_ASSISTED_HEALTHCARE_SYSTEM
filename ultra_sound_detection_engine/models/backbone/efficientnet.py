import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB3Backbone(nn.Module):
    def __init__(self, pretrained=True, in_channels=1):
        super().__init__()

        self.model = models.efficientnet_b3(pretrained=pretrained)

        # Modify input layer for grayscale ultrasound
        if in_channels == 1:
            self.model.features[0][0] = nn.Conv2d(
                1, 40, kernel_size=3, stride=2, padding=1, bias=False
            )

        self.features = self.model.features
        self.out_channels = 1536  # EfficientNet-B3 output channels

    def forward(self, x):
        return self.features(x)
