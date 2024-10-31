import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Classes / Methods
# -------------------------------------------------------------------------

class ConvNet(nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        #----------------------------------------
        # Convolutional Steps
        #----------------------------------------

        # Layer 1: 3 Channels (RGB), 3x3 kernel size, 32 kernels
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)  # 3 in channels for RGB
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Layer 2: 32 Channels, 3x3 kernel size, 64 kernels
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Layer 3: 64 Channels, 3x3 kernel size, 128 kernels
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #----------------------------------------
        # Connected Layers
        #----------------------------------------

        # Flatten (for connected layers)
        self.flatten = nn.Flatten()

        # Layer 1: Input size = 80x80, 450 neurons
        self.dense1 = nn.Linear(128 * 80 * 80, 450)

        # Layer 2: Dropout
        self.dropout = nn.Dropout(0.5)

        # Layer 3: Input size = 450, 450 neurons
        self.dense2 = nn.Linear(450, 450)

        # Output layers
        self.class_out = nn.Linear(450, num_classes)
        self.bbox_out = nn.Linear(450, 4)

    # Use Dropout now.
    def forward(self, X):
        # ======================================================================
        # Three convolutional layers + two fully connected layers
        # Uses ReLU and Dropout.
        # ======================================================================
        
        # First convolution pass
        X = self.conv1(X)
        X = F.relu(X)
        X = self.pool1(X)

        # Second convolution pass
        X = self.conv2(X)
        X = F.relu(X)
        X = self.pool2(X)

        # Third convolution pass
        X = self.conv3(X)
        X = F.relu(X)
        X = self.pool3(X)

        # Flatten the tensor
        X = self.flatten(X)

        # First fully connected layer
        X = self.dense1(X)
        X = F.relu(X)

        # Apply dropout
        X = self.dropout(X)

        # Second fully connected layer
        X = self.dense2(X)
        X = F.relu(X)

        # Get class and bbox outputs
        class_X = self.class_out(X)
        bbox_X = self.bbox_out(X)

        # Output layer
        return class_X, bbox_X