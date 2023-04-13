import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import gc
from datetime import datetime
import subprocess

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

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
        
def calc_label(series,treshold,steps):
    target = series.rolling(steps).mean().shift(-steps) /series.rolling(steps).mean().shift(steps)
    target = pd.DataFrame(target)
    target_col = series.name
    target['label'] = 1
    target['label'][target[target_col] <= (1 - treshold)]= 0
    target['label'][target[target_col] >= (1 + treshold)] = 2
    w = target.groupby('label').count().reset_index()
    if len(w) < 3:
        labels = [0,1,2]
        for l in labels:
            if l not in w['label'].values:
                w_ = pd.DataFrame([[l,0]])
                w_.columns = w.columns
                w = pd.concat([w,w_])

    return target['label'],w.sort_values('label').values[:,-1]



    return target['label'],target.groupby('label').count().values

class HourParquetDataset(Dataset):
    def __init__(self, paths : list,seq_len=100,stats = [],
                 mode='train',scale=True,clip=False,clip_values=[],
                 resample=False,clip_percent=1,
                 target_col=None,target_steps=5,treshold=1e-4,target_col_idx=-1,**kwargs):
        self.seq_len = seq_len
        self.target_col_idx = target_col_idx
        #read datasets
        # self.dsets = [pd.read_parquet(p) for p in paths]

        

        if resample:
            self.dsets = [pd.read_parquet(p,engine='fastparquet').resample(resample).last() for p in paths]
            print(f'{str(datetime.now())} : Data resampled.')
        else:
            self.dsets = [pd.read_parquet(p,engine='fastparquet') for p in paths]
            print(f'{str(datetime.now())} : Data loaded.')

        gc.collect()

        if target_col and target_steps:
            # targets = [calc_label(d[target_col],treshold,target_steps) for d in self.dsets]
            targets = Parallel(n_jobs=30)(delayed(calc_label)(d[target_col],treshold,target_steps) for d in self.dsets)
            self.targets = [t[0] for t in targets]
            weights = [t[1] for t in targets]
            weights = np.sum(weights,0)
            self.weights = (weights.sum() / weights / len(weights)).reshape(-1)
            self.dsets = [d[2*target_steps-1 : -target_steps] for d in self.dsets]
            print(f'{str(datetime.now())} : {mode} weights {self.weights}')
            del targets
        else:
            self.weights = None

        sizes = [len(d) - seq_len for d in self.dsets]
        self.folds = torch.cumsum(torch.tensor(sizes),0)

        if clip:
            self.clip_values = clip_values[[f'{clip_percent}min',f'{clip_percent}max']].values.T

            def clip_(d):
                return d.clip(lower=self.clip_values[0],upper=self.clip_values[1],axis=1)
            
            self.dsets = [clip_(d) for d in self.dsets]

            print(f'{str(datetime.now())} : Data clipped.')
        else:
            self.clip_values  = None
        gc.collect()

        if scale:
            self.stats = [[d.min(),d.max()] for d in self.dsets]

            #norm datasets
            for i in range(len(self.stats) -1):
                self.dsets[i+1] = (self.dsets[i+1] - self.stats[i][0] ) / (self.stats[i][1] - self.stats[i][0] + 1e-10)

            if len(stats) == 1:
                self.stats[0] = stats[-1]


            self.dsets[0] = (self.dsets[0] - self.stats[0][0] ) / (self.stats[0][1]  - self.stats[0][0] + 1e-10)
            print(f'{str(datetime.now())} : Data scaled.')
        
        gc.collect()




    def __len__(self):
        return self.folds[-1]

    def __getitem__(self, idx):
        fold = choose_fold(self.folds,idx)

        if fold > 0:
            idx -= self.folds[fold-1].item()

        X = self.dsets[fold][idx:idx+self.seq_len]
        if type(self.weights) != type(None):
            y = self.targets[fold][idx+self.seq_len]

            return torch.from_numpy(X.values).float().T,torch.Tensor([y]).long()[0]
        else:
            return torch.from_numpy(X.values).float().T
