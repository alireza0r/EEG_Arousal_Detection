import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import math

class Evaluator():
  def __init__(self, encoder, cluster, dataset, start_epoch=100, num_epochs=20):
    self.encoder = encoder
    self.cluster = cluster
    self.dataset = dataset
    self.channel = None
    self.start_epoch = start_epoch
    self.num_epochs = num_epochs

  def GenerateLatent(self, signal):
    if not isinstance(signal, torch.Tensor):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      signal = torch.tensor(signal, device=device, dtype=torch.float)

    self.encoder(signal)
    result = self.encoder.latent.detach().cpu().numpy()
    return result

  def SignalInitializer(self, start_epoch=None, num_epochs=None):
    if num_epochs:
      self.num_epochs = num_epochs
    if start_epoch:
      self.start_epoch = start_epoch

    if self.num_epochs>1:
      data, label = self.dataset.__getitem__(np.arange(self.start_epoch, self.start_epoch+self.num_epochs))
      return data.detach().cpu().numpy(), label.detach().cpu().numpy()
    elif self.num_epochs==1:
      data, label = self.dataset.__getitem__(np.arange(self.start_epoch, self.start_epoch+self.num_epochs))
      data = np.expand_dims(data, axis=0)
      return data.detach().cpu().numpy(), label.detach().cpu().numpy()
    else:
      return None

  def transform(self, x):
    return np.mean(x, axis=-1)

  def display(self, signal, classes, true_classes, extra_label=None):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    e_len = signal.shape[-1]
    x = np.arange(signal.shape[0]*signal.shape[-1])
    fig, ax = plt.subplots(1,1, figsize=(15,10))
    y_loc = signal[:,self.channel,:].mean()
    y_min = signal[:,self.channel,:].min()
    y_max = signal[:,self.channel,:].max()
    for i, (d, c, t_c) in enumerate(zip(signal, classes, true_classes)):
      if (i+1)==signal.shape[0]:
        break

      color = colors[c]
      ax.plot(x[i*e_len:(i+1)*e_len], d[self.channel,:], color=color)
      ax.axvline(x[i*e_len], linestyle='--')
      ax.text(x=x[i*e_len]+2, y=y_min, fontsize=10, s=f'{int(t_c[-1])}')

      if extra_label:
        ax.text(x=x[i*e_len]+2, y=y_min+0.005, fontsize=10, s=extra_label[i])

    ax.text(x=2, y=y_max, s=f'Channel:{self.channel}')
    plt.tight_layout()
    # plt.legend()
    plt.show()

  def __call__(self, channel, start_epoch=None, num_epochs=None, extra_label=None):
    if num_epochs:
      self.num_epochs = num_epochs

    self.channel = channel
    signal, label = self.SignalInitializer(start_epoch=start_epoch, num_epochs=self.num_epochs)
    latent = self.GenerateLatent(signal)
    latent = self.transform(latent)
    classes = self.cluster(latent)
    print('Classes: ', end='')
    print(classes)
    self.display(signal, classes, label, extra_label=extra_label)


def csv_categorize(csv_info, label_key='excitation'):
  subs = csv_info['SubName'].values

  sub, count = np.unique(subs, return_counts=True)
  print('Subs name and counts:', sub, count)

  search_key = ['train1', 'train2', 'test']
  result = {}
  easy_result = {}
  hard_result = {}
  test_result_bci_easy = {}
  test_result_bci_hard = {}
  test_result_sham_easy = {}
  test_result_sham_hard = {}
  for s in sub:
    sub_info = csv_info[csv_info['SubName']==s].copy()
    for key in search_key:
      name_sub_info = sub_info[sub_info['Name'].str.contains(key, regex=True)].copy()
      # print(s, key)
      # print(name_sub_info)

      # Test
      if key in search_key[-1]:
        bci_easy = []
        bci_hard = []
        sham_easy = []
        sham_hard = []
        # print(name_sub_info)
        for i in range(1, name_sub_info.shape[0]+1):
          if i%2 == 0: # Hard
            if i in [1,2,5,6]: # BCI
              bci_hard.append(name_sub_info[label_key].values[i-1])
            else:
              sham_hard.append(name_sub_info[label_key].values[i-1])

          else: # Easy
            if i in [1,2,5,6]: # BCI
              bci_easy.append(name_sub_info[label_key].values[i-1])
            else:
              sham_easy.append(name_sub_info[label_key].values[i-1])

        # Easy BCI
        if not (s in test_result_bci_easy.keys()):
          # easy_result[s] = [(key, np.array(easy).mean())]
          test_result_bci_easy[s] = [np.array(bci_easy).mean()]
        else:
          test_result_bci_easy[s].append(np.array(bci_easy).mean())

        # Hard BCI
        if not (s in test_result_bci_hard.keys()):
          # easy_result[s] = [(key, np.array(easy).mean())]
          test_result_bci_hard[s] = [np.array(bci_hard).mean()]
        else:
          test_result_bci_hard[s].append(np.array(bci_hard).mean())

        # Easy Sham
        if not (s in test_result_sham_easy.keys()):
          # easy_result[s] = [(key, np.array(easy).mean())]
          test_result_sham_easy[s] = [np.array(sham_easy).mean()]
        else:
          test_result_sham_easy[s].append(np.array(sham_easy).mean())

        # Hard Sham
        if not (s in test_result_sham_hard.keys()):
          # easy_result[s] = [(key, np.array(easy).mean())]
          test_result_sham_hard[s] = [np.array(sham_hard).mean()]
        else:
          test_result_sham_hard[s].append(np.array(sham_hard).mean())

      # Train
      if key in search_key[:-1]:
        hard = []
        easy = []
        print(name_sub_info)
        for i in range(1, name_sub_info.shape[0]+1):
          if i%2 == 0: # Hard
            hard.append(name_sub_info[label_key].values[i-1])
          else: # Easy
            easy.append(name_sub_info[label_key].values[i-1])

        if not (s in easy_result.keys()):
          # easy_result[s] = [(key, np.array(easy).mean())]
          easy_result[s] = [np.array(easy).mean()]
        else:
          easy_result[s].append(np.array(easy).mean())

        if not (s in hard_result.keys()):
          hard_result[s] = [np.array(hard).mean()]
        else:
          hard_result[s].append(np.array(hard).mean())

      if not (s in result.keys()):
        result[s] = [name_sub_info[label_key].values.mean()]
      else:
        result[s].append(name_sub_info[label_key].values.mean())

  result, easy_result, hard_result, test_result_bci_easy, test_result_bci_hard, test_result_sham_easy, test_result_sham_hard


def radar_chart(data, labels, title, label_transform=None, figsize=(10, 10)):

  variables = len(data[list(data.keys())[0]])

  # Calculate angles for each variable
  angles = np.linspace(0, 2 * math.pi, variables, endpoint=False)

  # Close the plot to create a circular shape
  # data1.append(data1[0])
  # data2.append(data2[0])
  # data3.append(data3[0])

  labels.append(labels[0])
  angles = np.concatenate((angles, [angles[0]]))

  # Create a figure and axes
  fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})

  # Plot the data
  for k in data.keys():
    data[k].append(data[k][0])

    if label_transform:
      ax.plot(angles, data[k], 'o-', linewidth=1, label=label_transform[k])
    else:
      ax.plot(angles, data[k], 'o-', linewidth=1, label=k)

  # Add labels for each variable
  ax.set_thetagrids(angles * 180 / math.pi, labels)

  # Set title
  ax.set_title(title)
  fig.legend()

  # Show the plot
  plt.show()