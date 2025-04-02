import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Second convolutional layer
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 35)  # Output layer (35 classes: 0-9 + a-z)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.max_pool2d(x, 2, 2)  # Pooling after first convolution
        x = F.relu(self.conv2(x))  
        x = F.max_pool2d(x, 2, 2)  # Pooling after second convolution
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Fully connected layer activation
        x = self.fc2(x)  # Output layer (no activation due to CrossEntropyLoss)
        return x
