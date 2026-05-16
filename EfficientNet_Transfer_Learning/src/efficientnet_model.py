import torch.nn as nn
from torchvision import models


class EfficientNetTransfer(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientNetTransfer, self).__init__()

        print("[MODEL] Loading EfficientNet-B0...")

        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Freeze backbone
        for param in self.model.features.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        print("[MODEL] Model ready with custom classifier")

    def forward(self, x):
        return self.model(x)