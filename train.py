from Libs.trainer import *
from Libs.utils import *
from Models.autoencoder import *
from Models.classification import *
from ExternalLibs.sdtw_cuda_loss import *
from Libs.layers import AvgGlobalPool2d, MaxGlobalPool2d, ModelFromJson

from tslearn.metrics import SoftDTWLossPyTorch
import torch
from torch.utils.data import Dataset, DataLoader
import torcheeg
# from torcheeg.transforms import MeanStdNormalize
import numpy as np
import mne
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
mne.set_log_level('WARNING')

import argparse
from time import time
from scipy.signal import spectrogram

import os

  
def training(args, dataloader, validation_loader, loss_fn, model):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Device:', device)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # Iterate through the dataloader
  for epoch in range(args.epochs):
    start_time = time()
    model.train()
    train_loss_list = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
      # inputs = inputs.unsqueeze(1)
      # inputs = inputs.permute(0, 2, 1)
      outputs = model(inputs.to(device))
      loss = loss_fn(outputs, labels.to(device))
      train_loss_list.append(loss.item())
      # print(loss)
      # print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    model.eval()
    loss_list = []
    r2_list = []
    for batch_idx, (inputs, labels) in enumerate(validation_loader):
      # inputs = inputs.unsqueeze(1)
      # inputs = inputs.permute(0, 2, 1)
      outputs = model(inputs.to(device))
      val_loss = loss_fn(outputs, labels.to(device))
      r2 = r2_score_loss(outputs, labels.to(device))
      loss_list.append(val_loss.item())
      r2_list.append(r2.item())
      # print(loss)
      # print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")
      
    print(f'Epoch {epoch+1}, Loss: {np.mean(train_loss_list):.3f}, Valid Loss: {np.mean(loss_list):.3f}, R2 Valid: {np.mean(r2_list):.3f}, Time: {time()-start_time:.3f} Sec.')

  torch.save(model.state_dict(), args.save+'.pth')

def stft_transform(fs, window_length, overlap):
    window_size = int(window_length * fs)
    overlap_size = int(window_size * overlap)
    def stft(eeg_signal):
      if len(eeg_signal.shape) == 2:
        sxx_list = []
        for ch in range(eeg_signal.shape[0]):
          f, t, Sxx = spectrogram(eeg_signal[ch,:], fs=fs, window='hamming', nperseg=window_size, noverlap=overlap_size)
          sxx_list.append(Sxx)
          # print(sxx_list)

        Sxx = np.stack(sxx_list, axis=0)
      else:
        f, t, Sxx = spectrogram(eeg_signal, fs=fs, window='hamming', nperseg=window_size, noverlap=overlap_size)
      return 20*np.log10(Sxx)
    return stft

def r2_score_loss(y_pred, y_true):
    # Calculate the residual sum of squares
    rss = torch.sum((y_true - y_pred)**2)
    
    # Calculate the total sum of squares
    tss = torch.sum((y_true - torch.mean(y_true))**2)
    
    # Calculate the R2 score
    r2 = 1 - (rss / tss)
    
    return r2

def model_v2(args, dataloader, validation_loader, loss_fn):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Device:', device)

  model = ModelFromJson(args.modelcon).Config()
  model.to(device)

  training(args, dataloader, validation_loader, loss_fn, model)
  
def train_classification_model(args, dataloader, validation_loader, loss_fn):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model = EEGNet(num_classes=1)
  model.to(device)

  training(args, dataloader, validation_loader, loss_fn, model)

  # loss_fn = nn.MSELoss()

  # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # # Iterate through the dataloader
  # for epoch in range(args.epochs):
  #   start_time = time()
  #   model.train()
  #   for batch_idx, (inputs, labels) in enumerate(dataloader):
  #     # inputs = inputs.unsqueeze(1)
  #     # inputs = inputs.permute(0, 2, 1)
  #     outputs = model(inputs.to(device))
  #     loss = loss_fn(outputs, labels.to(device))
  #     # print(loss)
  #     # print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")

  #     optimizer.zero_grad()
  #     loss.backward()
  #     optimizer.step()

  #   model.eval()
  #   loss_list = []
  #   for batch_idx, (inputs, labels) in enumerate(validation_loader):
  #     # inputs = inputs.unsqueeze(1)
  #     # inputs = inputs.permute(0, 2, 1)
  #     outputs = model(inputs.to(device))
  #     loss = loss_fn(outputs, labels.to(device))
  #     loss_list.append(loss.item())
  #     # print(loss)
  #     # print(f"Batch {batch_idx + 1}, Loss: {loss.item()}")
      
  #   print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Valid Loss: {np.mean(loss_list):.3f}, Time: {time()-start_time:.3f} Sec.')

  # torch.save(model.state_dict(), args.save+'.pth')


def auto_encoder_model(args, dataloader, validation_loader):
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

  parser.add_argument('--m', metavar='name', required=True, help='Model kind to train.', default='Classification', type=str)
  parser.add_argument('--datapath', metavar='path', required=True, help='EEG Epochs path (*-epo.fif file) or .npz data path for pre processing.', type=str)
  parser.add_argument('--h', metavar='Hz', required=False, help='High cut-off frequency for filter.', default=45, type=int)
  parser.add_argument('--l', metavar='Hz', required=False, help='Low cut-off frequency for filter.', default=1, type=int)
  parser.add_argument('--lr', metavar='float', required=False, help='Learning rate.', default=0.001, type=float)
  parser.add_argument('--epochs', metavar='int', required=False, help='Number of epochs.', default=200, type=int)
  parser.add_argument('--batch', metavar='int', required=False, help='Batch size.', default=8, type=int)
  parser.add_argument('--valid_split', metavar='float', required=False, help='Validation set.', default=0.2, type=float)
  parser.add_argument('--seed', metavar='int', required=False, help='Seed', default=42, type=int)
  parser.add_argument('--save', metavar='path', required=False, help='Path to save trained model weights', default='', type=str)
  parser.add_argument('--loss', metavar='name', required=False, help='Lass function name', default='MSE', type=str)
  parser.add_argument('--feature', metavar='name', required=False, help='Feature extraction mode', default='STFT', type=str)
  parser.add_argument('--winlen', metavar='value', required=False, help='window length in second', default=0.5, type=float)
  parser.add_argument('--overlap', metavar='value', required=False, help='Overlap lenght in second', default=0.25, type=float)
  parser.add_argument('--modelcon', metavar='path', required=False, help='YAML file to load into the model', default='./config.yaml', type=str)
  parser.add_argument('--p', required=False, help='Prepare data when it is not initialized as MNE epochs.', action='store_true')
  parser.add_argument('--condition', metavar='name', required=False, help='Conditions that .npz files should have to process.', default='', type=str)
  
  args = parser.parse_args()

  # Load dataset
  if not args.p: 
    print('Load MNE structured dataset')
    train_set = mne.read_epochs(args.datapath)
    ch = train_set.info['ch_names'][:-2]
    train_set.filter(l_freq=args.l, h_freq=args.h, picks=ch)
    train_features = train_set.copy()

    transform = None
    if args.feature == 'STFT':
      transform = stft_transform(fs=512, window_length=args.winlen, overlap=args.overlap)
    dataset = EEGDatasetV2(data=train_features.get_data()[:,:32,:],
                        label=train_features.events[:,-1],
                        transform=transform)

  else:
    print('Preparing data')
    data_list = []
    label_list = []
    for filename in os.listdir(args.datapath):
      if not (args.condition in filename):
        continue
        
      if filename.endswith(".npz"):  # Assuming data files are .npz files
          # Load the data
          data = np.load(os.path.join(args.datapath, filename))
          data_array = data['data']
          label_array = data['label'][:,1].astype('float')
          label_list.append(label_array)
  
          # Dataset
          data = mne.filter.filter_data(data=data_array, l_freq=1, h_freq=45, sfreq=512)
          data_list.append(data)

    data_list = np.concatenate(data_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    
    transform = None
    if args.feature == 'STFT':
      transform = stft_transform(fs=512, window_length=args.winlen, overlap=args.overlap)
    dataset = EEGDatasetV2(data=data_list,
                        label=label_list,
                        transform=transform)

    print('Data prepared')
    print(f'Data shape: {data_list.shape}, Label shape: {label_list.shape}')
    
  # train_features = mne.time_frequency.EpochsSpectrum(train_set, 
  #                                                    method='multitaper', 
  #                                                    fmin=0, 
  #                                                    fmax=args.h*2, 
  #                                                    tmin=0, 
  #                                                    tmax=9.998, 
  #                                                    picks=ch, 
  #                                                    proj=False, 
  #                                                    exclude='bads', 
  #                                                    n_jobs=2, 
  #                                                    remove_dc=True)
  # train_features = train_set.copy()

  

  # Use DataLoader to shuffle and batch the data
  batch_size = args.batch
  shuffle = True

  indices = list(range(dataset.size))
  if args.seed:
    np.random.seed(args.seed)
  np.random.shuffle(indices)
  split = int(np.floor(args.valid_split * dataset.size))
  train_indices, val_indices = indices[split:], indices[:split]
  
  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)
  # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, )
  train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
  validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

  # loss_fn = nn.MSELoss() if args.loss == 'MSE' else None
  # loss_fn = r2_score_loss if args.loss == 'R2' else None
  # assert loss_fn != None

  if args.loss == 'MSE':
    loss_fn = nn.MSELoss()
  elif args.loss == 'R2':
    loss_fn = r2_score_loss
  else:
    raise Exception('Unknown Loss function')
    

  if args.m == 'Classification':
    train_classification_model(args, train_loader, validation_loader, loss_fn)
  elif args.m == 'AutoEncoder':
    auto_encoder_model(args, train_loader, validation_loader)
  if args.m == 'Modelv2':
    model_v2(args, train_loader, validation_loader, loss_fn)
