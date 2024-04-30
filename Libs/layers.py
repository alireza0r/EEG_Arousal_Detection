import torch
import torch.nn as nn
import yaml

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

class ModelFromJson:
  def __init__(self, config_path):
    self.config_path = config_path
    self.config = self.LoadModel(config_path)

  def LoadModel(self, config_path):
    with open(config_path, 'r') as f:
      config = yaml.safe_load(f)
    return config['model']

  def Config(self):
    layers = []
    for l in self.config["layers"]:
      try:
        layer_type = getattr(nn, l["type"])
      except:
        layer_type = eval(l["type"])
      parameters = {key: value for key, value in list(l.items())[1:]}
      layer = layer_type(**parameters)
      layers.append(layer)
    return nn.Sequential(*layers)
