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

#train_data = data_processing.train_data
#val_data = data_processing.val_data
#utils.train(model, train_data, val_data, num_epochs=30, batch_size=256, lr = 0.01, name = "augmented_data")

model_augment = models.CNNClassifier()
model_unaugment = models.CNNClassifier()
noisy_test = utils.dataset_from_list(
    data_processing.test_data,
    transform=transforms.Compose([
        utils.add_noise_transform(snr_min=5, snr_max=10),
        utils.MyPipeline()
    ])
)

pitch_up_test = utils.dataset_from_list(
    data_processing.test_data,
    transform=transforms.Compose([
        utils.add_noise_transform(snr_min=5, snr_max=10),
        T.PitchShift(sample_rate = 8000, n_steps = -2),
        utils.MyPipeline()
    ])
)

pitch_down_test = utils.dataset_from_list(
    data_processing.test_data,
    transform=transforms.Compose([
        utils.add_noise_transform(snr_min=5, snr_max=10),
        T.PitchShift(sample_rate = 8000, n_steps = 2),
        utils.MyPipeline()
    ])
)

model_augment.load_state_dict(torch.load("Models/augmented_data_CNNClassifier_bs256_lr0.01_epoch25_val0.9547"))
model_unaugment.load_state_dict(torch.load("Models/CNNClassifier_bs64_lr0.01_epoch19_val0.9767"))

noisy_acc_clean_model = utils.get_accuracy(model_unaugment, noisy_test)
noisy_acc_aug_model   = utils.get_accuracy(model_augment, noisy_test)
pitch_up_acc_clean_model = utils.get_accuracy(model_unaugment, pitch_up_test)
pitch_down_acc_clean_model = utils.get_accuracy(model_unaugment, pitch_down_test)
pitch_up_acc_aug_model = utils.get_accuracy(model_augment, pitch_up_test)
pitch_down_acc_aug_model = utils.get_accuracy(model_augment, pitch_down_test)


print(f"Clean model on noisy audio : {noisy_acc_clean_model}, Pitch up: {pitch_up_acc_clean_model}, Pitch down: {pitch_down_acc_clean_model}")
print(f"Aug model on noisy audio : {noisy_acc_aug_model}, Pitch up: {pitch_up_acc_aug_model}, Pitch down: {pitch_down_acc_aug_model}")







