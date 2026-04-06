import time
import os
import numpy as np
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim 

import utils
import models
import data_processing
import constants

import torchaudio
import torchaudio.transforms as T

train_data = data_processing.train_data

val_data = data_processing.val_data

model = models.CNNGRUClassifier(hidden_size=128)


utils.train(model, train_data, val_data, num_epochs=20, batch_size=32, lr = 0.001, name = "")






