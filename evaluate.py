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
  parser = argparse.ArgumentParser(description='Evaluate AutoEncoder and Classification model')

  parser.add_argument('--m', metavar='name', required=True, help='Model kind to train.', default='Classification', type=str)
  parser.add_argument('--datapath', metavar='path', required=True, help='EEG Epochs path (*-epo.fif file).', type=str)
  parser.add_argument('--h', metavar='Hz', required=False, help='High cut-off frequency for filter.', default=45, type=int)
  parser.add_argument('--l', metavar='Hz', required=False, help='Low cut-off frequency for filter.', default=1, type=int)
  parser.add_argument('--batch', metavar='int', required=False, help='Batch size.', default=16, type=int)
  parser.add_argument('--load', metavar='path', required=True, help='Path for load trained model weights.', default='', type=str)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGNet(num_classes=1)
    model.load_state_dict(torch.load(args.load))
    model.to(device)
    model.eval()

    loss_fn = nn.MSELoss()
    loss_list = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
      # inputs = inputs.unsqueeze(1)
      # inputs = inputs.permute(0, 2, 1)
      outputs = model(inputs.to(device))
      loss = loss_fn(outputs, labels.to(device))
      # print(loss)
      # print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")

      loss_list.append(loss.item())
      # print(loss.item())

    print('Total Loss:', np.mean(loss_list))
  elif args.m == 'AutoEncoder':
    print('Model is not defined')