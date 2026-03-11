import time
import os
import numpy as np
from torch.utils import data
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim 

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from torch.utils.data import Dataset, DataLoader
import data_processing

def get_accuracy(model, type='train'):
    if type == 'train':
        dataset = data_processing.train_loader
    elif type == 'val':
        dataset = data_processing.val_loader
    elif type == 'test':
        dataset = data_processing.test_loader
    else:
      print('Wrong type, train, val, or test')
      return

    correct = 0
    total = 0
    for imgs, labels in dataset:


        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          imgs = imgs.cuda()
          labels = labels.cuda()
        #############################################


        output = model(imgs)

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train(model, data, batch_size=64, num_epochs=1, lr = 0.01):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    iters, losses, train_acc = [], [], []
    val_accuracies = []

    # training
    n = 0 # the number of iterations
    start_time=time.time()
    for epoch in range(num_epochs):
        mini_b=0
        for imgs, labels in iter(train_loader):


            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
            #############################################


            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            ##### Mini_batch Accuracy ##### We don't compute accuracy on the whole training set in every iteration!
            pred = out.max(1, keepdim=True)[1]
            mini_batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            Mini_batch_total = imgs.shape[0]
            train_acc.append((mini_batch_correct / Mini_batch_total))
           ###########################

          # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            n += 1
            mini_b += 1
            #print("Iteration: ",n,'Progress: % 6.2f ' % ((epoch * len(train_loader) + mini_b) / (num_epochs * len(train_loader))*100),'%', "Time Elapsed: % 6.2f s " % (time.time()-start_time))


        print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)))
        val = get_accuracy(model, 'val')
        val_accuracies.append(val)
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(model.name, batch_size, lr, epoch, val)
        torch.save(model.state_dict(), model_path)


    # Prepare val_acc for plotting
    val_acc_for_plot = []
    num_iterations_per_epoch = len(train_loader)
    for epoch_idx in range(num_epochs):
        val_acc_for_plot.extend([val_accuracies[epoch_idx]] * num_iterations_per_epoch)

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc_for_plot, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_accuracies[-1]))


#gotten from https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

#gotten from https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    power_to_db = T.AmplitudeToDB("power", 80.0)
    ax.imshow(power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

#gotten from https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

#modifed from https://docs.pytorch.org/audio/stable/transforms.html
class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        n_fft=512,
        n_mel=64,
        stretch_factor=0.8,
    ):
        super().__init__()
        self.target_samples = 8000 # 1 second at 8kHz
        
        self.spec = T.Spectrogram(n_fft=n_fft, power=2)

        self.spec_aug = torch.nn.Sequential(
            T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(freq_mask_param=15),
            T.TimeMasking(time_mask_param=35),
        )

        self.amplitude_to_db = T.AmplitudeToDB()

        self.mel_scale = T.MelScale(
            n_mels=n_mel, sample_rate=8000, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Pad or truncate to 8k samples
        if waveform.shape[1] > self.target_samples:
            waveform = waveform[:, :self.target_samples]
        elif waveform.shape[1] < self.target_samples:
            padding = self.target_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))

        # Convert to power spectrogram
        spec = self.spec(waveform)

        # Apply SpecAugment
        spec = self.spec_aug(spec)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        output = self.amplitude_to_db(mel)

        return output
    
class DatasetFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        label = int(file_name.split('_')[0])
        
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, label