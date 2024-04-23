import torch
import torch.nn as nn

class EEGNet(nn.Module):
  def __init__(self, num_classes=1, dropout=0.25):
    super(EEGNet, self).__init__()

    self.lstm1 = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True)
    self.cnn1 = nn.Conv1d(32, 32, 6, 3)
    self.batchnorm1 = nn.BatchNorm1d(32)
    self.act1 = nn.LeakyReLU()
    self.maxpool1 = nn.MaxPool1d(5,5)

    self.lstm2 = nn.LSTM(input_size=32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True)
    self.cnn2 = nn.Conv1d(32, 32, 6, 3)
    self.batchnorm2 = nn.BatchNorm1d(32)
    self.act2 = nn.LeakyReLU()
    self.maxpool2 = nn.MaxPool1d(5,5)

    self.dropout1 = nn.Dropout(p=dropout)
    self.fc1 = nn.Linear(32, 16)
    self.act3 = nn.LeakyReLU()
    self.fc2 = nn.Linear(16, num_classes)
    self.act4 = nn.ReLU()

  def forward(self, x):
    x, _ = self.lstm1(x.permute(0, 2, 1))
    x = self.cnn1(x.permute(0, 2, 1))
    x = self.batchnorm1(x)
    x = self.act1(x)
    x = self.maxpool1(x)

    x, _ = self.lstm2(x.permute(0, 2, 1))
    x = self.cnn2(x.permute(0, 2, 1))
    x = self.batchnorm2(x)
    x = self.act2(x)
    x = self.maxpool2(x)

    x = self.dropout1(x)
    x = torch.mean(x, -1)
    x = self.fc1(x)
    x = self.act3(x)
    x = self.fc2(x)
    x = self.act4(x)
    return x