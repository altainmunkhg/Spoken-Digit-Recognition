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
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        output = self.layer2(x)
        return output
    
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.name = "RNNClassifier"
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.layer1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.layer1(out[:, -1, :])
        return out

class CNNClassifierv2(nn.Module):
    def __init__(self):
        super(CNNClassifierv2, self).__init__()
        self.name = "CNNClassifier_v2"
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  
        self.pool  = nn.MaxPool2d(2, 2)
        self.layer1 = nn.Linear(64 * 8 * 4, 256)     
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        return self.layer2(x)
    
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUClassifier, self).__init__()
        self.name = "GRUClassifier"
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.layer1 = nn.Linear(hidden_size,10)

    def forward(self, x):
        x = x.squeeze(1)  
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.layer1(out[:, -1, :])
        return out
    
class CNNGRUClassifier(nn.Module):
    def __init__(self, n_mels=64, hidden_size=128, num_layers=2):
        super(CNNGRUClassifier, self).__init__()
        self.name = "CNNGRUClassifier"

        # CNN 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=(2,1))
        self.drop  = nn.Dropout(0.2)
        self.bn1   = nn.BatchNorm2d(16)
        self.bn2   = nn.BatchNorm2d(32)
        self.bn3   = nn.BatchNorm2d(64)

        # GRU
        self.gru_input_size = 64 * (n_mels // 8)
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        self.drop2 = nn.Dropout(0.2)
        self.fc    = nn.Linear(hidden_size * 2, 10) 
        self.instance_norm = nn.InstanceNorm2d(1)

    def forward(self, x):
        x = self.instance_norm(x)

        # CNN
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)

        # Reshape for GRU 
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2)      
        x = x.reshape(B, W, C * H)   

        x, _ = self.gru(x)           
        x = x[:, -1, :]        

        x = self.drop2(x)
        return self.fc(x) 