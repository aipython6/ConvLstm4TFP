import numpy as np
data_path = './train.npz.npy'
data = np.load(data_path)
print(data.sum(axis=3).sum(axis=2).sum(axis=1).sum(axis=0))