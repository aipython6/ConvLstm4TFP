import scipy.io
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

def poly(data):
    sns.heatmap(data,cmap='rainbow')
    plt.savefig('./test5.jpg', dpi = 400, bbox_inches='tight')
    plt.show()

def data():
    a = [
        {'idx': 1, 'in': [i for i in range(1,10)], 'out': [i for i in range(11,20)]},
        {'idx': 2, 'in' :[i for i in range(31,40)], 'out': [i for i in range(51,60)]},
        {'idx': 3, 'in' :[i for i in range(31,40)], 'out': [i for i in range(121,130)]},
        {'idx': 4, 'in' :[i for i in range(61,70)], 'out': [i for i in range(41,50)]},
        {'idx': 5, 'in' :[i for i in range(101,110)], 'out': [i for i in range(91,100)]},
        {'idx': 6, 'in' :[i for i in range(87,96)], 'out': [i for i in range(21,30)]},
    ]
    time_interval = []
    for index in range(0, len(a[0]['in'])):
        region = list()
        for item in a:
            region.append([item['in'][index],item['out'][index]])
        time_interval.append(region)
    result = np.array(time_interval)
    return result
def get_test():
    data = np.load('E:\\article\\trafficFlowPredictionConvLstm\\data\\test.npz.npy')
    arr = data[585]
    poly(arr[:,:,0])

# get_test()

def get_train():
    data_path = 'E:\\article\\trafficFlowPredictionConvLstm\\NYC-stdn\\volume_test.npz'
    data =  np.load(data_path)['volume']
    print(data)
    # data =  np.load(data_path)['volume'][56]
    data =  np.load(data_path)['volume']
    print(data)

    # poly(data[:,:,0])
get_train()
