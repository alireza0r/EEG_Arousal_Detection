import torch
from torch.utils.data import Dataset
import scipy
import os
import numpy as np
import mne
import pandas as pd

# from tslearn.metrics import SoftDTWLossPyTorch
# import torch
# from torch.utils.data import Dataset, DataLoader
import torcheeg
from torcheeg.transforms import MeanStdNormalize
# import numpy as np
# import mne
# import torch
# from torch.utils.data.sampler import SubsetRandomSampler

def csv_spliter(csv, condictions):
  new_csv = csv.copy()
  for k, v in zip(condictions.keys(), condictions.values()):
    if v == '':
      continue
    new_csv = new_csv[new_csv[k]==v].copy()
  print(f'New csv size: {new_csv.shape}')
  return new_csv

class EEGDatasetV2(Dataset):
    def __init__(self, data, label, transform=None):
        self.size = data.shape[0]
        self.data = data
        self.labels = label
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.transform:
            d = self.transform(self.data[idx])
            return torch.FloatTensor(d), torch.FloatTensor([self.labels[idx]])
        else:           
            return torch.FloatTensor(self.data[idx]), torch.FloatTensor([self.labels[idx]])

# from torch.nn.utils.rnn import pad_sequence, pack_sequence
class EEGDataset(Dataset):
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

# from torch.nn.utils.rnn import pad_sequence, pack_sequence
class MNEDataset(Dataset):
  def __init__(self, data, transform=None, normalize=True):
    super().__init__()

    self.data = data

    self.max = []
    self.min = []
    self.mean = []
    self.std = []
    for i in range(len(self.data.info['ch_names'])):
      self.max.append(data.get_data(i).max())
      self.min.append(data.get_data(i).min())
      self.mean.append(data.get_data(i).mean())
      self.std.append(data.get_data(i).std())

    self.transform = transform
    self.normalize = normalize

  def normal(self, data):
    return (data - np.array(self.mean).reshape((len(self.mean), 1))) / np.array(self.std).reshape((len(self.std), 1))

  def __len__(self):
    return self.data.get_data().shape[0]

  def __getitem__(self, idx):
    sample = self.data.get_data()[idx,]
    label = self.data.events[idx,]

    if self.transform:
      sample = self.transform(sample)

    # sample = MeanStdNormalize(axis=0)(eeg=sample)['eeg']

    if self.normalize:
      sample = self.normal(sample)

    return torch.Tensor(sample), torch.Tensor(label)

# eeg_dataset = MNEDataset(epochs)
# data, label = eeg_dataset.__getitem__(np.arange(100,200))


def load_ced_info(ced_path):
  # Read the .ced file into a pandas DataFrame
  ced_data = pd.read_csv(ced_path, delimiter='\t')

  # Extract electrode labels and coordinates
  electrode_names = ['TIME']+ced_data['labels'].tolist()+['REF', 'PPG', 'GSR', 'TRIGGER']
  electrode_coordinates = ced_data[['X', 'Y', 'Z']].values  # Assuming X, Y, Z columns represent coordinates
  electrode_coordinates = np.concatenate((np.zeros((1,3)), electrode_coordinates, np.zeros((4,3))), axis=0)


  # Create an MNE-Python Info structure
  info = mne.create_info(ch_names=electrode_names, sfreq=512, ch_types='eeg')
  info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(electrode_names, electrode_coordinates))), on_missing='warn')

  misc_channels = ['REF', 'PPG', 'GSR', 'TIME']
  for channel_name in misc_channels:
      info.set_channel_types({channel_name: 'misc'})

  info.set_channel_types({'TRIGGER': 'stim'})
  return info


class EEGPreProcessing():
  def __init__(self, files_path, info, event_name=None, find_key=''):
    self.files_path = files_path
    self.find_key = find_key
    self.info = info
    self.event_name = event_name

    self.last_raw = None

    self.files_list = self.FindWavFiles(files_path, find_key)
    assert len(self.files_list)>0, print('There was not any .wav files')
    # print(files)

  def FindWavFiles(self, dir, keyword, f_type='mat'):
    files_list = []
    for path, _, files in os.walk(dir):
      for f in files:
        if keyword in f:
          if f.split('.')[-1]==f_type:
            files_list.append(os.path.join(path, f))
    return files_list

  def LoadFiles(self, path, info=None):
    # Load the .mat file
    mat_data = scipy.io.loadmat(path)
    # print(mat_data.keys())

    raw = mat_data['data']
    raw = np.squeeze(raw)

    if info:
      raw = mne.io.RawArray(raw, self.info)

    self.last_raw = raw
    return raw

  def Filtering(self, data, picks):
    # return data.filter(1, 45, picks=picks)
    return data.filter(l_freq=None, h_freq=5, picks=picks) #, fir_design='firwin')

  def GenerateEvents(self, data, id=20, duration=5.5, stim_channel='TRIGGER', **arg):
    def find_overlap(events):
      """Find pairs of overlapping events"""
      overlap_pairs = []
      for i, event_i in enumerate(events):
          for event_j in events[i + 1:]:
              # Check if the events overlap in time
              if event_i[0] < event_j[0] < event_i[0] + event_i[2] or \
                event_j[0] < event_i[0] < event_j[0] + event_j[2]:
                  overlap_pairs.append((event_i, event_j))
      return overlap_pairs

    start = 10 # sec
    stop = data.times[-1] - 10 # sec

    fix_events = mne.make_fixed_length_events(data, id=id, start=start, stop=stop, duration=duration)
    events = mne.find_events(data, stim_channel=stim_channel)
    combined_events = np.concatenate([events, fix_events])

    # Find events overlap
    overlap_pairs = find_overlap(combined_events)
    # Print the overlapping event pairs
    print("Overlapping event pairs:")
    for pair in overlap_pairs:
      print(pair)

    # Removal overlap
    removal_list = []
    for j, e in enumerate(combined_events):
      for i in range(len(overlap_pairs)):
        if all(overlap_pairs[i][1] == e):
          print(j, i, e, overlap_pairs[i][1])
          removal_list.append(j)

    # Remove repitative events
    keep = list(range(len(combined_events)))
    for r in removal_list:
      keep.remove(r)
      print(f"Event {r} was removed.")

    combined_events = combined_events[keep]
    print(f"New event len: {len(combined_events)}")
    return combined_events

  def CountTrigger(self, value, stim_channel='TRIGGER'):
    if self.last_raw:
      signal = self.last_raw.get_data(stim_channel)[0]
      count = 0
      faling_flag=False
      rise_flag=False
      level_flag=False

      for i in range(len(signal)-1):
        if signal[i+1]>signal[i]:
          rise_flag=True
          faling_flag=False
          level_flag=False
        elif signal[i+1]<signal[i]:
          faling_flag=True
          rise_flag=False
          level_flag=False
        else:
          if signal[i]==value and rise_flag==True and faling_flag==False:
            if level_flag==False:
              count += 1
              level_flag=True
      return count
    else:
      return None

  def CountTriggerInPath(self, stim_channel='TRIGGER'):
    result = pd.DataFrame([], columns=['FileName', 'TriggerValue', 'TriggerCount'])
    for i, f in enumerate(self.files_list):
      raw = self.LoadFiles(f, self.info)
      values_in_file = np.unique(raw.get_data(stim_channel))
      try:
        count=[]
        for v in values_in_file:
          count.append(self.CountTrigger(value=v, stim_channel=stim_channel))
        result.loc[i] = [os.path.split(f)[-1], values_in_file, count]
      except:
        print(f'{i}: File {f} has problem to count')
    return result


  def __call__(self, tmin, tmax, duration=5.5, sig_picks=None):
    epochs_list, events_list = [], []
    for f in self.files_list:
      eeg_data = self.LoadFiles(f, self.info)
      print(eeg_data.get_data().shape)
      # raw = self.Filtering(eeg_data, picks=sig_picks)

      events = self.GenerateEvents(eeg_data, duration=duration)
      epochs = mne.Epochs(eeg_data,
                          events,
                          tmin=tmin,
                          tmax=tmax,
                          event_id=self.event_name,
                          preload=True,
                          picks=sig_picks)
      
      epochs = self.Filtering(epochs, picks=sig_picks)

      epochs_list.append(epochs)
      events_list.append(events)
    return mne.concatenate_epochs(epochs_list), np.concatenate(events_list, axis=0)


# Generate data and Events
def find_vote(voting_details, events):
  vote_list = []
  for e in events[:,0]:
    vote_list.append([-1, -1, -1])
    for t0, t1, v in zip(voting_details['StartSample'], voting_details['StopSample'], voting_details[['pleasure',	'excitation',	'involving']].values):
      if e>t0 and e<=t1:
        vote_list[-1] = v # Update value
        break
        # print(t0,t1,e)
  if len(vote_list) != events.shape[0]:
    print('Warning len(vote_list) != events.shape[0]', len(vote_list), '!=', events.shape[0])
  return vote_list


def find_label_from_csv(csv_info, index, keys=['StartSample', 'StopSample']):
  for i, (s, f) in enumerate(csv_info[keys]):
    if s<=index and index>=f:
      return csv_info.values[i]
  return None


class EEGPreProcessingV2():
  def __init__(self, files_path, info, event_name=None, voting_details=None, find_key=''):
    self.files_path = files_path
    self.find_key = find_key
    self.info = info
    self.event_name = event_name
    self.voting_details = voting_details

    self.last_raw = None

    self.files_list = self.FindWavFiles(files_path, find_key)
    assert len(self.files_list)>0, print('There was not any .wav files')
    # print(files)

  def FindWavFiles(self, dir, keyword, f_type='mat'):
    files_list = []
    for path, _, files in os.walk(dir):
      for f in files:
        if keyword in f:
          if f.split('.')[-1]==f_type:
            files_list.append(os.path.join(path, f))
    return files_list

  def LoadFiles(self, path, info=None):
    # Load the .mat file
    mat_data = scipy.io.loadmat(path)
    # print(mat_data.keys())

    raw = mat_data['data']
    raw = np.squeeze(raw)

    if info:
      raw = mne.io.RawArray(raw, self.info)

    self.last_raw = raw
    return raw

  def Filtering(self, data, picks):
    # return data.filter(1, 45, picks=picks)
    return data.filter(l_freq=None, h_freq=5, picks=picks) #, fir_design='firwin')

  def GenerateEvents(self, data, id=20, duration=5.5, stim_channel='TRIGGER', **arg):
    def find_overlap(events):
      """Find pairs of overlapping events"""
      overlap_pairs = []
      for i, event_i in enumerate(events):
          for event_j in events[i + 1:]:
              # Check if the events overlap in time
              if event_i[0] < event_j[0] < event_i[0] + event_i[2] or \
                event_j[0] < event_i[0] < event_j[0] + event_j[2]:
                  overlap_pairs.append((event_i, event_j))
      return overlap_pairs

    start = 10 # sec
    stop = data.times[-1] - 10 # sec

    fix_events = mne.make_fixed_length_events(data, id=id, start=start, stop=stop, duration=duration)
    events = mne.find_events(data, stim_channel=stim_channel)
    combined_events = np.concatenate([events, fix_events])

    # Find events overlap
    overlap_pairs = find_overlap(combined_events)
    # Print the overlapping event pairs
    print("Overlapping event pairs:")
    for pair in overlap_pairs:
      print(pair)

    # Removal overlap
    removal_list = []
    for j, e in enumerate(combined_events):
      for i in range(len(overlap_pairs)):
        if all(overlap_pairs[i][1] == e):
          print(j, i, e, overlap_pairs[i][1])
          removal_list.append(j)

    # Remove repitative events
    keep = list(range(len(combined_events)))
    for r in removal_list:
      keep.remove(r)
      print(f"Event {r} was removed.")

    combined_events = combined_events[keep]
    print(f"New event len: {len(combined_events)}")
    return combined_events

  def CountTrigger(self, value, stim_channel='TRIGGER'):
    if self.last_raw:
      signal = self.last_raw.get_data(stim_channel)[0]
      count = 0
      faling_flag=False
      rise_flag=False
      level_flag=False

      for i in range(len(signal)-1):
        if signal[i+1]>signal[i]:
          rise_flag=True
          faling_flag=False
          level_flag=False
        elif signal[i+1]<signal[i]:
          faling_flag=True
          rise_flag=False
          level_flag=False
        else:
          if signal[i]==value and rise_flag==True and faling_flag==False:
            if level_flag==False:
              count += 1
              level_flag=True
      return count
    else:
      return None

  def CountTriggerInPath(self, stim_channel='TRIGGER'):
    result = pd.DataFrame([], columns=['FileName', 'TriggerValue', 'TriggerCount'])
    for i, f in enumerate(self.files_list):
      raw = self.LoadFiles(f, self.info)
      values_in_file = np.unique(raw.get_data(stim_channel))
      try:
        count=[]
        for v in values_in_file:
          count.append(self.CountTrigger(value=v, stim_channel=stim_channel))
        result.loc[i] = [os.path.split(f)[-1], values_in_file, count]
      except:
        print(f'{i}: File {f} has problem to count')
    return result

  #
  def DatasetSpliter(self, data, condition:dict()):
    new_data = data.copy()
    for k in condition.keys():
      new_data = new_data[new_data[k]==condition[k]]
    return new_data

  def GenerateEventsFromCSV(self,
                            csv_info,
                            duration=10.0,
                            overlap=5.0,
                            timekeys=['StartSample', 'StopSample'],
                            labelkeys=['excitation'],
                            SR=512):

    duration, overlap = int(duration*SR), int(overlap*SR)
    label_list = []
    index_list = []
    for i, (s, f) in enumerate(csv_info[timekeys].values):
      sig_len = int(f - s)
      win_number = 0
      if sig_len >= duration:
        n = (sig_len - overlap) / (duration - overlap)
        n = int(n)

        for k in range(n):
          label_list.append(csv_info[labelkeys].iloc[i].values)
          index_list.append(int(s)+win_number*(duration - overlap))
          win_number += 1
        # print(csv_info['Name'].iloc[i], s, f, n, csv_info[labelkeys].iloc[i].values)

    indexs = np.stack(index_list, 0).reshape(-1, 1)
    labels = np.stack(label_list, 0)
    events = np.concatenate((indexs, np.zeros((indexs.shape[0], 1), dtype=int), labels), -1)
    return events, labels, indexs


  def __call__(self, tmin, tmax, duration=5.5, sig_picks=None, labelkeys=['excitation']):
    epochs_list, events_list = [], []
    full_vote_label = {}
    for f in self.files_list:
      eeg_data = self.LoadFiles(f, self.info)
      # print(eeg_data.get_data().shape)
      # raw = self.Filtering(eeg_data, picks=sig_picks)

      # events = self.GenerateEvents(eeg_data, duration=duration)
      # events = mne.find_events(eeg_data, stim_channel='TRIGGER')
      data_split = self.DatasetSpliter(self.voting_details, condition={'Name': os.path.split(f)[-1]})
      if len(data_split)==0:
        print('There were not any data on', f)
        continue
      events, _, _ = self.GenerateEventsFromCSV(data_split, labelkeys=labelkeys)

      epochs = mne.Epochs(eeg_data,
                          events,
                          tmin=tmin,
                          tmax=tmax,
                          baseline=(0,0),
                          # event_id=self.event_name,
                          preload=True,
                          picks=sig_picks)

      # print(events.shape, epochs.get_data().shape)
      epochs = self.Filtering(epochs, picks=sig_picks)

      epochs_list.append(epochs)
      events_list.append(events)

      print(f, 'was done.')

    return mne.concatenate_epochs(epochs_list), np.concatenate(events_list, axis=0)


def generate_latent(trainer, data_loader):
  # All train latent code
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  latent_code = []
  for d_loader in data_loader:
    trainer.model(d_loader[0].to(device))
    latent_code.append(trainer.model.latent.detach().cpu().numpy())
  latent = np.concatenate(latent_code, axis=0)

  print(latent.shape)
  return latent

# Select specific category
def dataset_spliter(data, condition:dict()):
  new_data = data.copy()
  for k in condition.keys():
    new_data = new_data[new_data[k]==condition[k]]
  return new_data
# dataset_spliter(csv_info, {'Level':'easy', 'Type':'bci', 'SubName':'sub9'})

if __name__ == '__main__':
  mne.set_log_level('WARNING')

  info = load_ced_info('/content/MyDrive/MyDrive/Code_AR/DataVisualization/channellocation_32ch.ced')
  # print(info['ch_names'])
  
  path = '/content/MyDrive/MyDrive/data/sub7/sub7_train1_20_06_2023_15_25_54_0000.mat'
  event_dict = {
          "Hit": 7,
          "Start-Mission": 14,
          "End-Silince": 15,
          "None": 20,
      }
  
  preprocess = EEGPreProcessing(files_path, info, event_name=event_dict, find_key='train')
  
  # raw = preprocess.LoadFiles(path, info)
  # preprocess.GenerateEvents(raw)
  
  # Generate epochs and ...
  epochs, events = preprocess(tmin=-0.25, tmax=4.7485, sig_picks=info['ch_names'][34:36])
  print(epochs.get_data().shape)
  print(events.shape) 
  
  preprocess.CountTriggerInPath()
