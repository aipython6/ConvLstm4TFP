import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '../NYC-stdn/volume_train.npz'
data = np.load(data_path)['volume']

# 降低一个维度
data1 = data.reshape(1920,200,2)
# print(data1.shape)
sum = data1.sum(axis=2)
# print(data1)
# print(sum.sum(axis=1))
# temp = sum.sum(axis=1)
# print(sum.sum(axis=1).reshape(1920,1))
a1 = sum.sum(axis=1).reshape(1920,1)
print(a1)
# sns.heatmap(a1[97:144].T, annot=True, fmt="d")
# plt.show()