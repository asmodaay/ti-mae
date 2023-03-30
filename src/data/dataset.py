import torch
from torch.utils.data import Dataset


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