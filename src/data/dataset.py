import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def create_synth_data(n_samples,seq_len=100,n_features=50,alpha=200,betta=3):
    t = torch.linspace(0,1,steps=seq_len)
    t = t.repeat(n_samples,n_features,1).float()
    X = torch.cos(alpha*t) + torch.cos(alpha*t/2) + torch.cos(alpha*t/4) + betta*t + torch.rand_like(t)
    return X

class SynthDataset(Dataset):
    def __init__(self, n_samples,seq_len=100,n_features=50,alpha=200,betta=3):
        self.data = create_synth_data(n_samples,seq_len=seq_len,n_features=n_features,alpha=alpha,betta=betta)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


class SimpleParquetDataset(Dataset):
    def __init__(self, path,seq_len=100):
        self.seq_len = seq_len
        self.data = pd.read_parquet(path)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):

        X = self.data.iloc[idx : idx + self.seq_len]
        X = (X - X.mean()) /( X.std() + 1)

        return torch.from_numpy(X.values).float().T
    

def choose_fold(folds,idx):
    for i,f in enumerate(folds):
        if idx < f:
            return i

class HourParquetDataset(Dataset):
    def __init__(self, paths : list,seq_len=100,stats = [],
                 mode='train',scale=True,clip=False,clip_values=[],
                 resample=False,**kwargs):
        self.seq_len = seq_len

        #read datasets
        self.dsets = [pd.read_parquet(p) for p in paths]
        if resample:
            self.dsets = [d.resample(resample).last() for d in self.dsets]

        sizes = [len(d) - seq_len for d in self.dsets]
        self.folds = torch.cumsum(torch.tensor(sizes),0)

        if clip:
            if not(len(clip_values)):
                self.clip_values = [
                    np.mean([d.quantile(0.01) for d in self.dsets],0),
                    np.mean([d.quantile(0.99) for d in self.dsets],0),
                ]
            else:
                self.clip_values = clip_values

            self.dsets = [d.clip(lower=self.clip_values[0],
                                 upper=self.clip_values[1],axis=1) for d in self.dsets]

        if scale:
            self.stats = [[d.min(),d.max()] for d in self.dsets]

            #norm datasets
            for i in range(len(self.stats) -1):
                self.dsets[i+1] = (self.dsets[i+1] - self.stats[i][0] ) / (self.stats[i][1] - self.stats[i][0] + 1e-10)

            if len(stats) == 1:
                self.stats[0] = stats[-1]


            self.dsets[0] = (self.dsets[0] - self.stats[0][0] ) / (self.stats[0][1]  - self.stats[0][0] + 1e-10)




    def __len__(self):
        return self.folds[-1]

    def __getitem__(self, idx):
        fold = choose_fold(self.folds,idx)

        if fold > 0:
            idx -= self.folds[fold-1].item()

        X = self.dsets[fold][idx:idx+self.seq_len]
        return torch.from_numpy(X.values).float().T
