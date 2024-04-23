from Libs.trainer import *
from Libs.utils import *
from Models.autoencoder import *
from Models.classification import *
from ExternalLibs.sdtw_cuda_loss import *

# from tslearn.metrics import SoftDTWLossPyTorch
import torch
from torch.utils.data import Dataset, DataLoader
# import torcheeg
# from torcheeg.transforms import MeanStdNormalize
import numpy as np
import mne
import torch
# from torch.utils.data.sampler import SubsetRandomSampler
import warnings
mne.set_log_level('WARNING')
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train AutoEncoder and Classification model')

  parser.add_argument('--m', metavar='name', required=True, help='Model kind to train.', default='Classification')
  parser.add_argument('--datapath', metavar='path', required=True, help='EEG Epochs path (*-epo.fif file).')
  parser.add_argument('--h', metavar='Hz', required=False, help='High cut-off frequency for filter.', default=1)
  parser.add_argument('--l', metavar='Hz', required=False, help='Low cut-off frequency for filter.', default=45)
  parser.add_argument('--lr', metavar='float', required=False, help='Learning rate.', default=0.0008)
  parser.add_argument('--epochs', metavar='int', required=False, help='Number of epochs.', default=200)
  parser.add_argument('--batch', metavar='int', required=False, help='Batch size.', default=200)
  parser.add_argument('--save', metavar='path', required=False, help='Path to save trained model weights', default='')
  args = parser.parse_args()

  # Load dataset
  train_set = mne.read_epochs(args.datapath)
  ch = train_set.info['ch_names'][:-2]
  train_set.filter(l_freq=args.l, h_freq=args.h, picks=ch)
  train_features = mne.time_frequency.EpochsSpectrum(train_set, 
                                                     method='multitaper', 
                                                     fmin=0, 
                                                     fmax=args.h*2, 
                                                     tmin=0, 
                                                     tmax=9.998, 
                                                     picks=ch, 
                                                     proj=False, 
                                                     exclude='bads', 
                                                     n_jobs=2, 
                                                     remove_dc=True)

  dataset = EEGDatasetV2(data=train_features.get_data(),
                      label=train_features.events[:,-1])

  # Use DataLoader to shuffle and batch the data
  batch_size = args.batch
  shuffle = True
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

  if args.m == 'Classification':
    model = EEGNet(num_classes=1)
    loss_fn = nn.MSELoss()
    loss_list = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
      # inputs = inputs.unsqueeze(1)
      # inputs = inputs.permute(0, 2, 1)
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      # print(loss)
      # print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")

      loss_list.append(loss.item())
      # print(loss.item())

    print('Total Loss:', np.mean(loss_list))
  elif args.m == 'AutoEncoder':
    print('Model is not defined')