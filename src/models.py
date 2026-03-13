import numpy as np
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F


class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.name = "ANNClassifier"
        self.fc1 = nn.Linear(64*32, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten feature data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.name = "CNNClassifier"
        self.conv1 = nn.Conv2d(1, 5, 5) #in_channels, out_chanels, kernel_size
        self.pool = nn.MaxPool2d(2, 2) #kernel_size, stride
        self.conv2 = nn.Conv2d(5, 10, 5) #in_channels, out_chanels, kernel_size
        self.layer1 = nn.Linear(13*5*10, 100)
        self.layer2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 13*5*10)
        x = F.relu(self.layer1(x))
        output = self.layer2(x)
        return output
