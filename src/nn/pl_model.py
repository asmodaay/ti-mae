import os
from torch import optim, nn, utils, Tensor
from src.nn.model import MaskedAutoencoder
import lightning.pytorch as pl



# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self,      
        in_chans=50,
        seq_len = 100,
        embed_dim=64,
        depth=2,
        num_heads=16,
        decoder_embed_dim=32,
        decoder_depth=2,
        decoder_num_heads=4,
        dropout = 0.1,
        norm_first = True,
        trunc_init = False,
        d_hid = 128,
        kernel_size = 3,
        stride = 1,
        padding =1,):

        super().__init__()
        self.model = MaskedAutoencoder(in_chans,seq_len,embed_dim,depth,num_heads,
                                       decoder_embed_dim,decoder_depth,decoder_num_heads,
                                       dropout,norm_first,trunc_init,d_hid,kernel_size,stride,padding)

    def training_step(self, batch, batch_idx):

        loss, pred, mask = self.model(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):

        loss, pred, mask = self.model(batch)
        self.log("eval_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
