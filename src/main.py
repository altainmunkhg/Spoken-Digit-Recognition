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



#model = models.RNNClassifier(hidden_size=128, input_size=64, num_classes=10)
model = models.ANNClassifier()
if constants.use_cuda and torch.cuda.is_available():
  model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')

print(len(data_processing.train_data))
train_data = data_processing.train_data
print((train_data[0][0].shape))
val_data = data_processing.val_data
#utils.train(model, train_data, val_data, num_epochs=20, batch_size=64, lr = 0.01)

model.load_state_dict(torch.load("Models/ANNClassifier_bs64_lr0.01_epoch17_val0.7567"))
print (f"Val Acc: {utils.get_accuracy(model, data_processing.test_data):.4f}")





