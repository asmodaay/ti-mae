from src.nn.pl_model import LitAutoEncoder
from src.data.dataset import HourParquetDataset

from torch import optim, nn, utils, Tensor

import lightning.pytorch as pl


autoencoder = LitAutoEncoder(in_chans=75)

paths = ['/workspace/TI-MAE/2022-12-02_13:00:00.parquet',
         '/workspace/TI-MAE/2022-12-09_13:00:00.parquet',
         '/workspace/TI-MAE/2022-12-13_13:00:00.parquet']

eval_paths = ['/workspace/TI-MAE/2022-12-14_18:00:00.parquet']

train_dataset = HourParquetDataset(paths,100)
eval_dataset = HourParquetDataset(eval_paths,100,stats=[train_dataset.stats[-1]])

train_loader = utils.data.DataLoader(train_dataset,64,num_workers=16)
eval_loader = utils.data.DataLoader(eval_dataset,64,num_workers=16)


trainer = pl.Trainer(max_epochs=10,val_check_interval = 5000)
trainer.fit(autoencoder,train_loader,eval_loader)