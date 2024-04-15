from sklearn.cluster import DBSCAN
import numpy as np

class DBSCANCluster:
  def __init__(self, eps=1, min_samples=5):
    self.eps = eps
    self.min_samples = min_samples
    
    self.cluster = None

  def train(self, x):
    self.cluster = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    self.cluster.fit(x)
    return self.cluster

  def save(self, path):
    # Save the DBSCAN clustering results 
    with open(path+'.pkl', 'wb') as f:
      pickle.dump(self.cluster, f)

  def load(self, path):
    # Save the DBSCAN clustering results 
    with open(path, 'rb') as f:
      self.cluster = loaded_dbscan = pickle.load(f)

  def __call__(self, x):
    return self.cluster.fit_predict(x)

if __name__ == '__main__':
  dcscan = DBSCANCluster()
  dcscan.train(x)          # Train
  result = dcscan(x_test)  # Test
  print(result)
