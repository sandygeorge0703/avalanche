# Import necessary libraries
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network with EWC strategy
    """

    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Conv Layer 1
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),           # Conv Layer 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Max Pooling 1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # Conv Layer 3
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),           # Conv Layer 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Max Pooling 2
            nn.Conv2d(64, 64, kernel_size=1, padding=0),           # Conv Layer 5
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),                               # Adaptive Pooling
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)                             # Classifier
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
