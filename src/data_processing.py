import utils
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T


data_dir = 'Free Spoken Digit Dataset (FSDD)/recordings'
data = utils.DatasetFolder(data_dir, transform=utils.MyPipeline())

#85% training, 10% validation, 5% testing
train_size = int(0.85*len(data))
val_size = int(0.10*len(data))
test_size = len(data) - train_size - val_size

#split up the data into the diffrent datas
train_data, val_data, test_data = torch.utils.data.random_split(data, [train_size,val_size, test_size])

#print out some stats
print('Num total recordings: ', len(data))
print('Num training recordings: ', len(train_data))
print('Num validation recordings: ', len(val_data))
print('Num testing recordings: ', len(test_data))

#train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
#val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

