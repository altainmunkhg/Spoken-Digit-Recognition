import utils
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import numpy as np



data_dir = 'Free Spoken Digit Dataset (FSDD)/recordings'


unmodified_data = utils.dataset_from_file(data_dir)



#split up the data into the diffrent datas
training_unaugmented_size = int (0.5*len(unmodified_data))
val_size = int(0.25*len(unmodified_data))
test_size = len(unmodified_data)  - val_size - training_unaugmented_size
training_unaugmented_data, val_data, test_data = torch.utils.data.random_split(unmodified_data, [training_unaugmented_size, val_size, test_size])


#data augmentation
heavy_noise_data = utils.dataset_from_list(training_unaugmented_data, transform= transforms.Compose([
                                                            utils.add_speed_change_transform(speed_min=0.9, speed_max=1.1),
                                                            utils.add_noise_transform(snr_min=10, snr_max = 20),
                                                            utils.MyPipeline(train=True)
                                                            ]))
light_noise_data = utils.dataset_from_list(training_unaugmented_data, transform= transforms.Compose([
                                                            utils.add_speed_change_transform(speed_min=0.9, speed_max=1.1),
                                                            utils.add_noise_transform(snr_min=3, snr_max = 10),
                                                            utils.MyPipeline(train=True)
                                                            ]))
pitch_up_data = utils.dataset_from_list(training_unaugmented_data, transform= transforms.Compose([
                                                            utils.add_speed_change_transform(speed_min=0.9, speed_max=1.1),
                                                            T.PitchShift(sample_rate=8000, n_steps=6),
                                                            utils.MyPipeline(train=True)
                                                            ]))
pitch_down_data = utils.dataset_from_list(training_unaugmented_data, transform= transforms.Compose([
                                                            utils.add_speed_change_transform(speed_min=0.9, speed_max=1.1),
                                                            T.PitchShift(sample_rate=8000, n_steps=-6),
                                                            utils.MyPipeline(train=True)
                                                            ]))
pitch_down_noisy_data = utils.dataset_from_list(training_unaugmented_data, transform= transforms.Compose([
                                                            utils.add_speed_change_transform(speed_min=0.9, speed_max=1.1),
                                                            T.PitchShift(sample_rate=8000, n_steps=-6),
                                                            utils.add_noise_transform(snr_min=3, snr_max = 10),
                                                            utils.MyPipeline(train=True)
                                                            ]))
pitch_up_noisy_data = utils.dataset_from_list(training_unaugmented_data, transform= transforms.Compose([
                                                            utils.add_speed_change_transform(speed_min=0.9, speed_max=1.1),
                                                            T.PitchShift(sample_rate=8000, n_steps=6),
                                                            utils.add_noise_transform(snr_min=3, snr_max = 10),
                                                            utils.MyPipeline(train=True)
                                                            ]))

training_unaugmented_data = utils.dataset_from_list(training_unaugmented_data, transform=utils.MyPipeline(train=True))
val_data = utils.dataset_from_list(val_data, transform=utils.MyPipeline(train=True))
test_data = utils.dataset_from_list(test_data, transform=None)


#utils.save_to_files(heavy_noise_data, "src/augmentedData/heavy_noise_data")
#utils.save_to_files(light_noise_data, "src/augmentedData/light_noise_data")
#utils.save_to_files(pitch_up_data, "src/augmentedData/pitch_up")
#utils.save_to_files(pitch_up_data, "src/augmentedData/pitch_up")
#utils.save_to_files(pitch_down_data, "src/augmentedData/pitch_down")
#utils.save_to_files(pitch_up_noisy_data, "src/augmentedData/pitch_up_noisy")
#utils.save_to_files(pitch_down_noisy_data, "src/augmentedData/pitch_down_noisy")

train_data = torch.utils.data.ConcatDataset([
    training_unaugmented_data,
    utils.retreive_from_file("src/augmentedData/heavy_noise_data"),
    utils.retreive_from_file("src/augmentedData/light_noise_data"),
    utils.retreive_from_file("src/augmentedData/pitch_up"), 
    utils.retreive_from_file("src/augmentedData/pitch_down"),
    utils.retreive_from_file("src/augmentedData/pitch_down_noisy"),
    utils.retreive_from_file("src/augmentedData/pitch_up_noisy")
])
#print out the stats
print('Num augmented recordings: ', len(train_data))
print('Num unaugmented recording: ', len(unmodified_data))
print('Num training recordings: ', len(train_data))
print('Num validation recordings: ', len(val_data))
print('Num testing recordings: ', len(test_data))

#train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
#val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

