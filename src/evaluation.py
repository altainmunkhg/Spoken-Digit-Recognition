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

#models being elavuated
model_augment = models.CNNClassifier()
model_unaugment = models.CNNClassifier()
model_pitch_augment = models.CNNClassifierv2()

model_augment.load_state_dict(torch.load("Models/augmented_data_CNNClassifier_bs256_lr0.01_epoch25_val0.9547"))
model_unaugment.load_state_dict(torch.load("Models/CNNClassifier_bs64_lr0.01_epoch19_val0.9767"))
model_pitch_augment.load_state_dict(torch.load("Models/pitched_data_CNNClassifier_v2_bs256_lr0.01_epoch19_val0.9773"))


#the test datasets being run
clean_test = utils.dataset_from_list(
    data_processing.test_data,
    transform=utils.MyPipeline()
)

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
        T.PitchShift(sample_rate = 8000, n_steps = 2),
        utils.MyPipeline()
    ])
)

pitch_down_test = utils.dataset_from_list(
    data_processing.test_data,
    transform=transforms.Compose([
        utils.add_noise_transform(snr_min=5, snr_max=10),
        T.PitchShift(sample_rate = 8000, n_steps = -2),
        utils.MyPipeline()
    ])
)


#getting the accuracy of the models
clean_acc_clean_model = utils.get_accuracy(model_unaugment, clean_test)
noisy_acc_clean_model = utils.get_accuracy(model_unaugment, noisy_test)
pitch_up_acc_clean_model = utils.get_accuracy(model_unaugment, pitch_up_test)
pitch_down_acc_clean_model = utils.get_accuracy(model_unaugment, pitch_down_test)

clean_acc_augment_model = utils.get_accuracy(model_augment, clean_test)
noisy_acc_aug_model   = utils.get_accuracy(model_augment, noisy_test)
pitch_up_acc_aug_model = utils.get_accuracy(model_augment, pitch_up_test)
pitch_down_acc_aug_model = utils.get_accuracy(model_augment, pitch_down_test)

clean_acc_pitch_model = utils.get_accuracy(model_pitch_augment, clean_test)
noisy_acc_pitch_model   = utils.get_accuracy(model_pitch_augment, noisy_test)
pitch_up_acc_pitch_model = utils.get_accuracy(model_pitch_augment, pitch_up_test)
pitch_down_acc_pitch_model = utils.get_accuracy(model_pitch_augment, pitch_down_test)


#print results
print(f"Clean model on clean audio: {clean_acc_clean_model: .4f}, noisy audio : {noisy_acc_clean_model: .4f}, Pitch up: {pitch_up_acc_clean_model: .4f}, Pitch down: {pitch_down_acc_clean_model: .4f}")
print(f"Aug model on clean audio: {clean_acc_augment_model: .4f}, noisy audio : {noisy_acc_aug_model: .4f}, Pitch up: {pitch_up_acc_aug_model: .4f}, Pitch down: {pitch_down_acc_aug_model: .4f}")
print(f"Pitch model on clean audio: {clean_acc_pitch_model: .4f}, noisy audio : {noisy_acc_pitch_model: .4f}, Pitch up: {pitch_up_acc_pitch_model: .4f}, Pitch down: {pitch_down_acc_pitch_model: .4f}")


#self_recorded_data = utils.dataset_from_file('self_recorded/Altai', transform=utils.MyPipeline())
#self_recorded_acc = utils.get_accuracy(model_augment, self_recorded_data)
#print(f"Self recorded acc: {self_recorded_acc: .4f}")
#print (utils.get_accuracy_by_class(model_augment, self_recorded_data))




