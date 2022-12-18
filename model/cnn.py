import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,in_channel,out_channels,hidden_size,output_size,drop_prob):
        super(CNN,self).__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channels[0], kernel_size=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.dropout = nn.Dropout(drop_prob)
        # nn.Flatten()
        self.fc = nn.Linear(hidden_size,output_size)


    def forward(self,x):
        x = x.view(x.shape[0],x.shape[1],2,10,20)
        cnn_feats = self.convs(x.transpose(1,2))
        # print(cnn_feats.shape)
        fusion_feats = cnn_feats.view(cnn_feats.shape[0],-1)
        x = self.fc(self.dropout(fusion_feats))
        return F.sigmoid(x)

if __name__ == "__main__":
    pass