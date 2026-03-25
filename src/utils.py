import time
import os
from PIL.ImageChops import constant
import numpy as np
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim 

import torchaudio
import torchaudio.functional as FA
import torchaudio.transforms as TA

from torch.utils.data import Dataset, DataLoader
import constants

import multiprocessing


def get_accuracy(model, data):
    correct = 0
    total = 0
    for recording, labels in torch.utils.data.DataLoader(data, batch_size=64):


        #############################################
        #To Enable GPU Usage
        if constants.use_cuda and torch.cuda.is_available():
          recording = recording.cuda()
          labels = labels.cuda()
        #############################################


        output = model(recording)

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        #print(recording.shape[0])
        total += recording.shape[0]
    return correct / total

def get_loss(model, data, criterion):
    total_loss = 0.0
    with torch.no_grad():
        for recording, labels in torch.utils.data.DataLoader(data, batch_size=64):
            output = model(recording)
            loss = criterion(output, labels)
            total_loss += loss.item() * recording.size(0)  
    return total_loss / len(data)  

def get_accuracy_by_class(model, data):
    class_correct = [0] * 10
    class_total = [0] * 10
    for recording, labels in torch.utils.data.DataLoader(data, batch_size=64):
        output = model(recording)
        pred = output.max(1, keepdim=True)[1]
        correct_tensor = pred.eq(labels.view_as(pred))
        correct = correct_tensor.numpy()
        
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    return class_accuracy

def train(model, train_data,val_data, batch_size=64, num_epochs=1, lr = 0.01, name = "unnamed"):
    num_workers = multiprocessing.cpu_count() -1
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=num_workers,
                                               prefetch_factor=2,      
                                               persistent_workers=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5)

    iters, losses, train_acc = [], [], []
    val_accuracies = []

    # training
    n = 0 # the number of iterations
    best_val = 0
    start_time=time.time()
    for epoch in range(num_epochs):
        mini_b=0
        model.train()

        for recording, labels in iter(train_loader):
            if (n % 10 == 0):
                print("*", end = "")
            

            #############################################
            #To Enable GPU Usage
            if constants.use_cuda and torch.cuda.is_available():
              recording = recording.cuda()
              labels = labels.cuda()
            #############################################


            out = model(recording)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            ##### Mini_batch Accuracy ##### We don't compute accuracy on the whole training set in every iteration!
            pred = out.max(1, keepdim=True)[1]
            mini_batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            Mini_batch_total = recording.shape[0]
            train_acc.append((mini_batch_correct / Mini_batch_total))
           ###########################

          # save the current training information
            iters.append(n)
            losses.append(loss.item()/batch_size)             # compute *average* loss
            n += 1
            mini_b += 1
            #print("Iteration: ",n,'Progress: % 6.2f ' % ((epoch * len(train_loader) + mini_b) / (num_epochs * len(train_loader))*100),'%', "Time Elapsed: % 6.2f s " % (time.time()-start_time))

        model.eval()
        print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)))
        with torch.no_grad():
            val = get_accuracy(model, val_data)
        val_accuracies.append(val)
        print(f"Epoch {epoch}, Val Acc: {val:.4f}")
        if val > best_val:
            best_val = val
            model_path = "Models/{0}_{1}_bs{2}_lr{3}_epoch{4}_val{5:.4f}".format(name, model.name, batch_size, lr, epoch, val)
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved (val={val:.4f})")
        

    # Prepare val_acc for plotting
    val_acc_for_plot = []   
    num_iterations_per_epoch = len(train_loader)
    for epoch_idx in range(num_epochs):
        val_acc_for_plot.extend([val_accuracies[epoch_idx]] * num_iterations_per_epoch)

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
    power_to_db = TA.AmplitudeToDB("power", 80.0)
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
        
        self.spec = TA.Spectrogram(n_fft=n_fft, power=2)

        self.amplitude_to_db = TA.AmplitudeToDB()

        self.mel_scale = TA.MelScale(
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

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        output = self.amplitude_to_db(mel)

        # Normalize
        output = (output - output.mean()) / (output.std() + 1e-6)

        return output
    
class dataset_from_file(Dataset):
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
    
class dataset_from_list(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        waveform = self.data[0][idx]
        label = self.data[1][idx]
        
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, label
    
class data_To_RNN_Type(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, label = self.dataset[idx]
        # Reshape to (seq_len, input_size) for RNN
        waveform = waveform.squeeze(0).transpose(0, 1)  # (32, 64)
        return waveform, label

class add_noise_transform:
    def __init__(self, snr_min=5, snr_max = 20):
        #snr = signal to noise ratio
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __call__(self, waveform):
        snr_db = torch.FloatTensor(1).uniform_(self.snr_min, self.snr_max)
        noise = torch.randn_like(waveform)
        return TA.AddNoise()(waveform, noise, snr_db)
