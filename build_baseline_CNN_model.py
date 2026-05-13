import torch
import torch.nn as nn
from torchsummary import summary

# Sub-task 1: Thiết kế CNN architecture cơ bản
class SkinBaselineCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SkinBaselineCNN, self).__init__()
        
        # Feature Extraction: 3 block Conv + Pool đơn giản
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 28x28
        )
        
        # Classifier: Nhận đặc trưng và phân loại vào 8 lớp
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Chống Overfitting
            nn.Linear(512, num_classes)
        )

    # Sub-task 2: Define forward pass
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Sub-task 3: Kiểm tra model summary
model = SkinBaselineCNN(num_classes=8)
print(summary(model, (3, 224, 224), device="cpu"))