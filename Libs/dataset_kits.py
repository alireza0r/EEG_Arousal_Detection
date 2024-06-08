# import os
# print(os.getcwd())
import pandas as pd
from scipy.io import loadmat
import os
# from Libs.utils import csv_spliter
from utils import *
import conditions
import argparse
import numpy as np

# Create and Split dataset into sub datasets
def dataset_spliter(df, root, duration=30, overlap=0.1, sr=512):
  # Step 2: Extract file paths
  file_paths = df['Name'].tolist()

  # Define epoch duration (in samples, assuming a specific sampling rate)
  epoch_duration = duration * sr  # 10 seconds * sampling rate

  # Calculate overlap duration (in samples)
  overlap_samples = int(epoch_duration * overlap)

  mat_data = None
  last_path = ''
  epoch_list = []
  label_list = []
  # Step 3: Iterate through data, segment into epochs with overlap, and filter based on labels
  for path, start, stop, valence, arousal, dominance, sub_name, ct_gd, hints, group, t, level in zip(
    file_paths,
    df['StartSample'],
    df['StopSample'],
    df['Valence'],
    df['Arousal'],
    df['Dominance'],
    df['SubName'],
    df['CT-GD'],
    df['Hints'],
    df['Group'],
    df['Type'],
    df['Level']
  ):
    try:
      if path != last_path:
        p = os.path.join(root, path.split('_')[0], path)
        mat_data = loadmat(p)
      last_path = path

      signal = mat_data['data'][1:33, 0, :]  # Extract first 32 channels

      # Iterate through the signal with overlap and segment into epochs
      i = start
      while i + epoch_duration <= stop: #signal.shape[-1]:
        epoch_data = signal[:, i:i+epoch_duration]

        # Filter epochs based on label values
        # Example: If valence is positive, process the epoch
        condition = True
        if condition:
          # Do whatever you want with the epoch data
          epoch_list.append(epoch_data)
          label_list.append([valence, arousal, dominance, sub_name, ct_gd, hints, group, t, level])

          # print(f"Epoch extracted from {path} starting at sample {i}")
          # print(f"Labels: Valence={valence}, Arousal={arousal}, Dominance={dominance}, SubName={sub_name}, CT-GD={ct_gd}, Hints={hints}, Group={group}, Type={t}, Level={level}")

        # Move to the next epoch with overlap
        i += epoch_duration - overlap_samples
    except Exception as e:
      print(f"Error loading or processing {path}: {e}")

    print(path, 'was done.')

  return np.stack(epoch_list, 0), np.stack(label_list, 0)


if __name__=='__main__':
  parser = argparse.ArgumentParser(description ='Load csv info for spliter')
  
  parser.add_argument('--path', 
                      metavar='path', 
                      type=str, 
                      help='Path to save the .npz files.',
                      default='/content/MyDrive/MyDrive/data/')
  parser.add_argument('--save', 
                      metavar='path', 
                      type=str, 
                      help='Path to save the .npz files.',
                      default='./')
  
  args = parser.parse_args()
  
  full_data = []
  for s, n in zip(conditions.split_list, conditions.conditions):
    name = ''
    for k, v in zip(n.keys(), n.values()):
      if v == 'control ':
        vv = 'control'
      else:
        vv = v
      name += k+'_'+vv+'-'
    res = dataset_spliter(s, root=args.path)
    print(res[0].shape, res[1].shape)
    if args.save != '':
      np.savez_compressed(os.path.join(args.save, name+'.npz'), data=res[0], label=res[1])
      print(name+'.npy', 'was saved')
      del res
    else:
      full_data.append(res)
