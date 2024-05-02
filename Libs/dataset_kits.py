# import os
# print(os.getcwd())
import pandas as pd
from scipy.io import loadmat
import os
# from Libs.utils import csv_spliter
from utils import *
from ..Metadata.conditions import *
import argparse

# Create and Split dataset into sub datasets
def dataset_spliter(df, root, duration=10, overlap=0.5, sr=512):
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
  for path, start, valence, arousal, dominance, sub_name, ct_gd, hints, group, t, level in zip(
    file_paths,
    df['StartSample'],
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

      signal = mat_data['data'][:32, 0,:]  # Extract first 32 channels

      # Iterate through the signal with overlap and segment into epochs
      i = start
      while i + epoch_duration <= signal.shape[-1]:
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
                      type=int, 
                      help='Path to save the .npz files.',
                      defult='/content/MyDrive/MyDrive/data/full_details_last_modified.csv')
  
  args = parser.parse_args()
  
  
  for s, n in zip(split_list, conditions):
    name = ''
    for k, v in zip(n.keys(), n.values()):
      if v == 'control ':
        vv = 'control'
      else:
        vv = v
      name += k+'_'+vv+'-'
    res = dataset(s, root=root)
    np.savez(os.path.join(args.path, name+'.npz'), data=res[0], label=res[1])
    print(name+'.npy', 'was saved')
    del res