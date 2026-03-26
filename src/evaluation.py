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
        T.PitchShift(sample_rate=8000, n_steps=2),   
        utils.add_noise_transform(snr_min=5, snr_max=10),
        utils.MyPipeline()
    ])
)

pitch_down_test = utils.dataset_from_list(
    data_processing.test_data,
    transform=transforms.Compose([
        T.PitchShift(sample_rate=8000, n_steps=-2), 
        utils.add_noise_transform(snr_min=5, snr_max=10),
        utils.MyPipeline()
    ])
)

self_recorded_data = utils.dataset_from_file('self_recorded/Altai', transform=utils.MyPipeline())


test_datasets = {
    "Clean"        : clean_test,
    "Noisy"        : noisy_test,
    "Pitch Up"     : pitch_up_test,
    "Pitch Down"   : pitch_down_test,
    "Self Recorded": self_recorded_data,
}

eval_models = {
    "Augmented"       : model_augment,
    "Pitch Augmented" : model_pitch_augment,
    "Unaugmented"     : model_unaugment,
}

#getting the accuracy of the models as table
header = f"{'Dataset':<16}" + "".join(f"{name:>18}" for name in eval_models)
print(header)
print("-" * (16 + 18 * len(eval_models)))

for dataset_name, dataset in test_datasets.items():
    row = f"{dataset_name:<16}"
    for model_name, model in eval_models.items():
        acc  = utils.get_accuracy(model, dataset)
        row += f"{acc:>17.1%} "
    print(row)


