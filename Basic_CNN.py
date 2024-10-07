################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco, Antonio Carta                                  #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################


import torch.nn as nn

from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)

class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network

    **Example**::

        >>> from avalanche.models import SimpleCNN
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleCNN(num_classes=n_classes)
        >>> print(model)  # View model details
        SimpleCNN(
          (features): Sequential(
            (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
            (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Dropout(p=0.25, inplace=False)
            (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (7): ReLU(inplace=True)
            (8): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
            (9): ReLU(inplace=True)
            (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (11): Dropout(p=0.25, inplace=False)
            (12): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (13): ReLU(inplace=True)
            (14): AdaptiveMaxPool2d(output_size=1)
            (15): Dropout(p=0.25, inplace=False)
          )
          (classifier): Sequential(
            (0): Linear(in_features=64, out_features=10, bias=True)
          )
        )
    """


    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Conv Layer 1
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),           # Conv Layer 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Max Pooling 1
            nn.Dropout(p=0.25),                                    # Dropout
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # Conv Layer 3
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),           # Conv Layer 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Max Pooling 2
            nn.Dropout(p=0.25),                                    # Dropout
            nn.Conv2d(64, 64, kernel_size=1, padding=0),           # Conv Layer 5
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),                               # Adaptive Pooling
            nn.Dropout(p=0.25),                                    # Dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)                             # Classifier
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class MTSimpleCNN(SimpleCNN, MultiTaskModule):
    """
    Convolutional Neural Network
    with multi-head classifier
    """

    def __init__(self):
        super(MTSimpleCNN, self).__init__()
        self.classifier = MultiHeadClassifier(64)  # Multi-head classifier for different tasks

    def forward(self, x, task_labels):
        x = self.features(x)
        x = x.squeeze()  # Remove dimensions of size 1 (for multi-task setup)
        x = self.classifier(x, task_labels)  # Pass task labels for multi-task learning
        return x


# Ensuring __all__ includes both models for easy import
__all__ = ["SimpleCNN", "MTSimpleCNN"]

