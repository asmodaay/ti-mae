from src.nn.pl_model import LitAutoEncoder
from src.data.dataset import HourParquetDataset
import os
from torch import  utils

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from datetime import datetime
import sys 


SEQ_LEN = int(sys.argv[1])
BS = int(sys.argv[2])

autoencoder = LitAutoEncoder(in_chans=75,
                             seq_len=SEQ_LEN)

data_path = '/workspace/ti-mae/data'
paths = [os.path.join(data_path,p) for p in sorted(os.listdir(data_path))]

eval_paths = paths[-2:]
paths = paths[:-2]

print(f'{str(datetime.now())} : Creating train dataset.')
train_dataset = HourParquetDataset(paths,SEQ_LEN)

print(f'{str(datetime.now())} : Creating eval dataset.')

eval_dataset = HourParquetDataset(eval_paths,SEQ_LEN,stats=[train_dataset.stats[-1]])

train_loader = utils.data.DataLoader(train_dataset,BS,num_workers=16)
eval_loader = utils.data.DataLoader(eval_dataset,BS*4,num_workers=16)

callbacks = [ModelCheckpoint(save_top_k=2, monitor="eval_loss"),
             EarlyStopping(patience=10,monitor="eval_loss")]

trainer = pl.Trainer(max_epochs=10,val_check_interval = 5000,accelerator="gpu", 
                     devices=-1,gradient_clip_val = 20.,
                     callbacks = callbacks)
trainer.fit(autoencoder,train_loader,eval_loader)