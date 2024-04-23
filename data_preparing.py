import mne
mne.set_log_level('WARNING')

import argparse
from Libs.utils import *
import os
import pandas as pd

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate dataset')

  parser.add_argument('--rootpath', metavar='path', required=True, help='Root path for all EEG files')
  parser.add_argument('--cedpath', metavar='path', required=True, help='Path for *.ced file')
  parser.add_argument('--csvpath', metavar='path', required=False, help='Path for *.csv file, file to generate events and epochs.')
  parser.add_argument('--key', metavar='keyword', required=False, help='File will be selected if the keyword being in the file name.', default='')
  parser.add_argument('--savepath', metavar='path', required=False, help='Path to save epochs.', default='./')
  args = parser.parse_args()

  info = load_ced_info(args.cedpath)
  files_path = args.rootpath

  with open(args.csvpath, 'r') as f:
    csv_info = pd.read_csv(f, index_col=0)

  event_dict = {
                # "Hit": 7,
                # "Start-Mission": 14,
                # "End-Silince": 15,
               }

  preprocess = EEGPreProcessingV2(files_path,
                                  info, event_name=event_dict,
                                  find_key='_test',
                                  voting_details=csv_info)

  epochs, events = preprocess(tmin=0,
                              tmax=9.998,
                              sig_picks=info['ch_names'][1:33]+info['ch_names'][34:36])

  print('Epochs and events shapes:', epochs.get_data().shape, events.shape)

  save_path = os.path.join(args.savepath, args.key+'-epo.fif')
  epochs.save(save_path)
  print(f'File was generated and saved in: {save_path}')