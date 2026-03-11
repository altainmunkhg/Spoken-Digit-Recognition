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

import torchaudio
import torchaudio.transforms as T

waveform, sample_rate = torchaudio.load("Free Spoken Digit Dataset (FSDD)/recordings/0_george_0.wav")
mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
spectrogram = T.Spectrogram(n_fft=512)

# Perform transform
spec = spectrogram(waveform)
mel_spec = mel_spectrogram(waveform)


fig, axs = plt.subplots(3, 1)
utils.plot_waveform(waveform, sample_rate, title="Original waveform", ax=axs[0])
utils.plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
utils.plot_spectrogram(mel_spec[0], title="mel spectrogram", ax=axs[2])

fig.tight_layout()
plt.show()

