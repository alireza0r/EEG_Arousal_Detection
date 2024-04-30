import torch
import torch.nn as nn

class AvgGlobalPool2d(nn.Module):
  def __init__(self,):
    super().__init__()

  def forward(self, x):
    assert len(x.shape) >= 3, print(x.shape)
    return torch.mean(x, dim=(-2,-1))

class MaxGlobalPool2d(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    assert len(x.shape) >= 3, print(x.shape)
    return torch.amax(x, dim=(-2,-1))
