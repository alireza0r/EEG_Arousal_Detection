import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
from torch.utils.data import Dataset, DataLoader
import os
import mne

class AutoEncoder(nn.Module):
  def __init__(self, in_ch):
    super().__init__()

    # Encoder structure
    pooling_kernel = 4
    pooling_stride = 4
    self.EN_POOL = nn.MaxPool1d(pooling_kernel,pooling_stride, return_indices=True)

    conv = nn.Conv1d(in_ch, 64, 3, 1, padding='same')
    act = nn.Tanh()
    norm = nn.BatchNorm1d(64)
    self.EN_CNN1 = nn.Sequential(conv, act, norm)

    conv = nn.Conv1d(64, 128, 3, 1, padding='same')
    act = nn.Tanh()
    norm = nn.BatchNorm1d(128)
    self.EN_CNN2 = nn.Sequential(conv, act, norm)

    conv = nn.Conv1d(128, 256, 3, 1, padding='same')
    act = nn.Tanh()
    norm = nn.BatchNorm1d(256)
    self.EN_CNN3 = nn.Sequential(conv, act, norm)

    # Decoder structures
    self.DE_POOL = nn.MaxUnpool1d(pooling_kernel,pooling_stride)

    conv = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1)
    act = nn.Tanh()
    # up = nn.Upsample(scale_factor=2)
    norm = nn.BatchNorm1d(128)
    self.DE_CNN1 = nn.Sequential(conv, act, norm)

    conv = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
    act = nn.Tanh()
    # up = nn.Upsample(scale_factor=2)
    norm = nn.BatchNorm1d(64)
    self.DE_CNN2 = nn.Sequential(conv, act, norm)

    conv = nn.ConvTranspose1d(64, in_ch, kernel_size=3, stride=1, padding=1)
    act = nn.Tanh()
    # up = nn.Upsample(scale_factor=2)
    norm = nn.BatchNorm1d(in_ch)
    self.DE_CNN3 = nn.Sequential(conv, act) #, norm)

    # Keep latent code
    self.latent = None

    # Keep indexes
    self.indexes = [None, None, None]

  def Encoder(self, x):
    x = self.EN_CNN1(x)
    x, self.indexes[0] = self.EN_POOL(x)

    x = self.EN_CNN2(x)
    x, self.indexes[1] = self.EN_POOL(x)

    x = self.EN_CNN3(x)
    x, self.indexes[2] = self.EN_POOL(x)
    return x

  def Decoder(self, x):
    x = self.DE_POOL(x, self.indexes[2])
    # print(2, x.size())
    x = self.DE_CNN1(x)
    # print(3, x.size())

    x = self.DE_POOL(x, self.indexes[1])
    # print(4, x.size())
    x = self.DE_CNN2(x)
    # print(5, x.size())

    x = self.DE_POOL(x, self.indexes[0])
    # print(6, x.size())
    x = self.DE_CNN3(x)
    return x


  def forward(self, x):
    x = self.Encoder(x)
    # print(0, 'Latent code size:', x.size())
    self.latent = x
    x = self.Decoder(x) * 5.0
    # print(x.size())
    return x

if __name__ == '__main__':
  auto_encoder = AutoEncoder(32)
  auto_encoder(torch.zeros((8, 32, 1024)))
  print(auto_encoder.latent.shape)
