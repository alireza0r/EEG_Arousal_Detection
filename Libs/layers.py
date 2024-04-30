import torch
import torch.nn as nn

class AvgGlobalPool2d(nn.Module):
  def __init__(self,):
    super().__init__()

  def forward(self, x):
    return torch.mean(x, dim=[x.size(-2), x.size(-1)])

class MaxGlobalPool2d(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.max(x, dim=[x.size(-2), x.size(-1)])
  
