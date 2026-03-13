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



model = models.CNNClassifier()
if constants.use_cuda and torch.cuda.is_available():
  model.cuda()
  print('CUDA is available!  Training on GPU ...')
else:
  print('CUDA is not available.  Training on CPU ...')

print(len(data_processing.train_data))

#utils.train(model, data_processing.train_data, data_processing.val_data, num_epochs=20, batch_size=64, lr = 0.005)
model.load_state_dict(torch.load("Models/model_CNNClassifier_bs64_lr0.005_epoch16_val0.6700"))
print (f"Test Acc: {utils.get_accuracy(model, data_processing.val_data):.4f}")





