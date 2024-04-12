import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

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

  def display(self, signal, classes, true_classes):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    e_len = signal.shape[-1]
    x = np.arange(signal.shape[0]*signal.shape[-1])
    fig, ax = plt.subplots(1,1, figsize=(15,10))
    for i, (d, c, t_c) in enumerate(zip(signal, classes, true_classes)):
      if (i+1)==signal.shape[0]:
        break

      color = colors[c]
      print(np.min(d[self.channel,:]))
      print(np.max(d[self.channel,:]))
      ax.plot(x[i*e_len:(i+1)*e_len], d[self.channel,:], color=color)
      ax.axvline(x[i*e_len], linestyle='--')
      # ax.text(x=x[i*e_len]+2, y=np.sin(i*np.pi/2)*8, s=f'{t_c}')

    # ax.text(x=2, y=-90, s=f'Channel:{self.channel}')
    print('ch')
    # plt.tight_layout()
    # plt.legend()
    plt.show()

  def __call__(self, channel, start_epoch=None, num_epochs=None):
    if num_epochs:
      self.num_epochs = num_epochs

    self.channel = channel
    signal, label = self.SignalInitializer(start_epoch=start_epoch, num_epochs=self.num_epochs)
    latent = self.GenerateLatent(signal)
    latent = self.transform(latent)
    classes = self.cluster(latent)
    print('Classes:')
    print(classes)
    self.display(signal, classes, label)

if __name__ == '__main__':
  eval = Evaluator(trainer.model, KNN.predict, eeg_dataset, start_epoch=100, num_epochs=60)
  eval(channel=0)
