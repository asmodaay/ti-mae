import torch
from torch.utils.data import Dataset
import pandas as pd

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
    def __init__(self, paths : list,seq_len=100,stats = [],mode='train'):
        self.seq_len = seq_len

        #read datasets
        self.dsets = [pd.read_parquet(p) for p in paths]
        sizes = [len(d) - seq_len for d in self.dsets]
        self.folds = torch.cumsum(torch.tensor(sizes),0)
        
        self.stats = [[d.mean(),d.std()] for d in self.dsets]

        if len(stats) == 1:
            self.stats[0] = stats[-1]

        #norm datasets
        for i in range(len(self.stats) -1):
            self.dsets[i+1] = (self.dsets[i+1] - self.stats[i][0] ) / (self.stats[i][1] + 1 )

        self.dsets[0] = (self.dsets[0] - self.stats[0][0] ) / (self.stats[0][1] + 1 )



    def __len__(self):
        return self.folds[-1]

    def __getitem__(self, idx):

        fold = choose_fold(self.folds,idx)
        X = self.dsets[fold][idx:idx+self.seq_len]

        return torch.from_numpy(X.values).float().T