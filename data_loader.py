import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class NYCSTDNDataset():
    def __init__(self,data_path,window_size=7) -> None:
        self.window_size = window_size
        self.data = torch.from_numpy(self.loading(data_path)).float()

    def loading(self,data_path):
        # data = np.load(data_path)['volume']
        data = np.load(data_path)
        self.max_val,self.min_val = np.max(data),np.min(data)
        dataset = slidingWindow(data,self.window_size)
        dataset = np.array(dataset).transpose(0,1,4,2,3) # (1914, 7, 2, 10, 20)
        dataset = dataset.reshape(dataset.shape[0],dataset.shape[1],-1)
        dataset = (dataset - self.min_val) / (self.max_val - self.min_val)
        return dataset
     
    def denormalize(self,x):        
        return x * (self.max_val - self.min_val) + self.min_val

def slidingWindow(seqs,size):
    """
    seqs: ndarray sequence, shape(seqlen,area_nums,2)
    size: sliding window size
    """
    result = []
    for i in range(seqs.shape[0] - size + 1):
        result.append(seqs[i:i + size,:,:,:]) #(7, 10, 20, 2) 
    # print(np.array(result).shape)
    return result


if __name__ == "__main__":
    # train_path = 'NYC-stdn/volume_train.npz'
    train_path = 'data/train.npz.npy'

    dataset = NYCSTDNDataset(train_path)
    # print(dataset.data[:1].shape)
    # print(dataset.data.shape)
    # data = np.load(train_path)['volume']
    # t = data[0][0]
    # print(t.shape)
    # rs_dataset = data.reshape(1920,400)

    # sns.heatmap(rs_dataset, cmap='Spectral')
    # plt.show()
    # test_path = 'NYC-stdn/volume_test.npz'
    test_path = 'data/test.npz.npy'
    dataset = NYCSTDNDataset(test_path)
    print(dataset.data.shape)
    # print(dataset.data)
    # data = np.load(test_path)['volume']
    # print(data.shape)   # shape(960,10,20,2)
    # print(data[0].shape)  # shape(10,20,2)
    # print(data[0][0])
    # h_data = data.reshape(960,200,2)
    # print(h_data)
    # print(data[0][0][0])    #一个区域的流入流出量
