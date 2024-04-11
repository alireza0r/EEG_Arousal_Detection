from Libs.trainer import *
from Libs.utils import *
from Models.autoencoder import *
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

# from torch.nn.utils.rnn import pad_sequence, pack_sequence
class MNEDataset(Dataset):
  def __init__(self, data, transform=None):
    self.data = data

    self.mean = np.mean(data.get_data(), axis=-1)
    self.mean = np.mean(self.mean, axis=0)

    self.std = np.mean(data.get_data(), axis=-1)
    self.std = np.mean(self.std, axis=0)

    self.transform = transform

  def __len__(self):
    return self.data.get_data().shape[0]

  def __getitem__(self, idx):
    sample = self.data.get_data()[idx,]

    if self.transform:
      sample = self.transform(sample)

    sample = MeanStdNormalize(axis=1)(eeg=sample)['eeg']
    
    return torch.Tensor(sample)


if __name__ == '__main__':
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

  eeg_dataset = MNEDataset(epochs)
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
