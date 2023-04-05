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
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=2,
        decoder_num_heads=4,
        dropout = 0.1,
        norm_first = True,
        trunc_init = False,
        d_hid = 128,
        kernel_size = 3,
        stride = 1,
        padding =1,
        scale_mode = 'adaptive_scale',
        forecast_ratio=0.25,
        forecast_steps = 10):

        super().__init__()
        self.model = MaskedAutoencoder(in_chans,seq_len,embed_dim,depth,num_heads,
                                       decoder_embed_dim,decoder_depth,decoder_num_heads,
                                       dropout,norm_first,trunc_init,d_hid,kernel_size,stride,padding,
                                       scale_mode = scale_mode,
                                       forecast_ratio=forecast_ratio,forecast_steps=forecast_steps)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):

        loss, pred, mask = self.model(batch)
        loss_removed , loss_seen, forecast_loss, backcast_loss = loss

        loss = loss_removed + 0.75 * forecast_loss + 0.5 * loss_seen + 0.2 * backcast_loss

        self.log("train/loss_removed", loss_removed, sync_dist=True)
        self.log("train/loss_seen", loss_seen, sync_dist=True)
        self.log("train/forecast_loss", forecast_loss, sync_dist=True)
        self.log("train/backcast_loss", backcast_loss, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True)



        return loss
    
    def validation_step(self, batch, batch_idx):

        loss, pred, mask = self.model(batch)
        loss_removed , loss_seen, forecast_loss, backcast_loss = loss
        loss = loss_removed + 0.75 * forecast_loss + 0.5 * loss_seen + 0.2 * backcast_loss

        self.log("eval/loss_removed", loss_removed, sync_dist=True)
        self.log("eval/loss_seen", loss_seen, sync_dist=True)
        self.log("eval/forecast_loss", forecast_loss, sync_dist=True)
        self.log("eval/backcast_loss", backcast_loss, sync_dist=True)
        self.log("eval/loss", loss, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=1e-5,weight_decay=1e-5)
        return optimizer



