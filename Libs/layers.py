import torch
import torch.nn as nn

class AvgGlobalPool2d(nn.Module):
  def __init__(self, axis):
    super().__init__()
  
    self.axis = axis

  def forward(self, x):
    return torch.mean(x, dim=[x.size(-2), x.size(-1)])

class MaxGlobalPool2d(nn.Module):
  def __init__(self, axis):
    super().__init__()
  
    self.axis = axis

  def forward(self, x):
    return torch.max(x, dim=[x.size(-2), x.size(-1)])
  
