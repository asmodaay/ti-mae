from src.nn.pl_model import LitAutoEncoder
from src.data.dataset import SynthDataset

from torch import optim, nn, utils, Tensor

import lightning.pytorch as pl


autoencoder = LitAutoEncoder()

dataset = SynthDataset(100000,seq_len=100,n_features=50,alpha=200,betta=3)
eval_dataset = SynthDataset(1000,seq_len=100,n_features=50,alpha=200,betta=3)

train_loader = utils.data.DataLoader(dataset,64)
eval_loader = utils.data.DataLoader(eval_dataset,64)


trainer = pl.Trainer(max_epochs=10,val_check_interval = 100)
trainer.fit(autoencoder,train_loader,eval_loader)