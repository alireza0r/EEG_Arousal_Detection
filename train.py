from Libs.trainer import *
from Libs.utils import *
from Models.autoencoder import *
from Models.classification import *
from ExternalLibs.sdtw_cuda_loss import *

from tslearn.metrics import SoftDTWLossPyTorch
import torch
from torch.utils.data import Dataset, DataLoader
import torcheeg
from torcheeg.transforms import MeanStdNormalize
import numpy as np
import mne
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
mne.set_log_level('WARNING')

import argparse
  
def train_classification_model(args, dataloader):
  model = EEGNet(num_classes=1)
  loss_fn = nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # Iterate through the dataloader
  for epoch in range(args.epochs):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
      # inputs = inputs.unsqueeze(1)
      # inputs = inputs.permute(0, 2, 1)
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      # print(loss)
      # print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

  torch.save(model.state_dict(), args.save+'.pth')


def auto_encoder_model(args, dataloader):
  info = load_ced_info('/content/MyDrive/MyDrive/Code_AR/DataVisualization/channellocation_32ch.ced')
  # print(info['ch_names'])
  
  files_path = '/content/MyDrive/MyDrive/data/'
  event_dict = {
          "Hit": 7,
          "Start-Mission": 14,
          "End-Silince": 15,
          "None": 20,
      }
  
  preprocess = EEGPreProcessing(files_path, info, event_name=event_dict, find_key='train')
  
  # Generate epochs and ...
  epochs, events = preprocess(tmin=-0.25, tmax=4.7485, sig_picks=info['ch_names'][34:36])
  print(epochs.get_data().shape)
  print(events.shape)

  print('Number of each Trigger:')
  print(preprocess.CountTriggerInPath())

  # eeg_dataset = MNEDataset(epochs)
  eeg_dataset = EEGDataset(epochs)
  data_length = eeg_dataset.data.get_data().shape
  
  warnings.filterwarnings("ignore", ".*copy=False will change to copy=True in 1.7*")
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  seed = 42
  torch.manual_seed(seed)
  
  split = int(data_length[0]*0.70)
  indices = list(range(data_length[0]))
  train_indices, val_indices = indices[:split], indices[split:]
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  
  train_loader = DataLoader(eeg_dataset, batch_size=8, shuffle=False, sampler=train_sampler, num_workers=2, pin_memory=True)
  valid_loader = DataLoader(eeg_dataset, batch_size=8, shuffle=False, sampler=valid_sampler, num_workers=2, pin_memory=True)

  num_channel = 2
  auto_encoder = AutoEncoder(num_channel).to(device)
  # MSE = nn.MSELoss(reduction='mean')
  # Loss = nn.CrossEntropyLoss()
  # Loss = fastdtw
  Loss = SoftDTWLossPyTorch(gamma=0.1)
  # Loss = DTWLoss()
  
  optim = torch.optim.Adam(auto_encoder.parameters(), lr=0.0009)
  trainer = Trainer(auto_encoder, Loss, optim)
  
  trainer.train(train_loader=train_loader,
                val_loader=valid_loader,
                num_epochs=100,)


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
    train_classification_model(args, dataloader)
  elif args.m == 'AutoEncoder':
    auto_encoder_model(args, dataloader)