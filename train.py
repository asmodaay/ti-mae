from src.nn.pl_model import LitAutoEncoder
from src.data.dataset import HourParquetDataset
import os
from torch import  utils

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from datetime import datetime
from torchinfo import summary
import hydra

import warnings
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="/workspace/ti-mae/configs", config_name="base.yaml")
def main(cfg):
    autoencoder = LitAutoEncoder(**cfg.model)

    summary(autoencoder.model,(1,cfg.model.in_chans,cfg.model.seq_len))

    data_path = cfg.data.path
    paths = [os.path.join(data_path,p) for p in sorted(os.listdir(data_path))]

    eval_paths = paths[-cfg.data.eval_dataset.size:]
    paths = paths[:-cfg.data.eval_dataset.size]
    paths = paths[-cfg.data.train_dataset.size:]
    
    print(f'{str(datetime.now())} : Creating train dataset.')
    train_dataset = HourParquetDataset(paths, **cfg.data.train_dataset)
    print(f'{str(datetime.now())} : Train dataset size : {len(paths)} hours , {len(train_dataset)} samples.')
    print(f'{str(datetime.now())} : Creating eval dataset.')

    eval_dataset = HourParquetDataset(eval_paths,
                                      stats=[train_dataset.stats[-1]], 
                                      clip_values=train_dataset.clip_values,
                                      **cfg.data.eval_dataset)
    print(f'{str(datetime.now())} : Eval dataset size : {len(eval_paths)} hours , {len(eval_dataset)} samples.')


    train_loader = utils.data.DataLoader(train_dataset,cfg.data.train_batch_size,num_workers=cfg.data.loader_workers)
    eval_loader = utils.data.DataLoader(eval_dataset,cfg.data.eval_batch_size,num_workers=cfg.data.loader_workers)

    callbacks = [ModelCheckpoint(**cfg.callbacks.checkpointing),
                EarlyStopping(**cfg.callbacks.early_stopping)]

    trainer = pl.Trainer(callbacks = callbacks,**cfg.trainer)
    trainer.fit(autoencoder,train_loader,eval_loader,ckpt_path=cfg.ckpt_path)

if __name__ == "__main__":
    main()