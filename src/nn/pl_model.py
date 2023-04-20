import os
from torch import optim, save
from src.nn.model import MaskedAutoencoder
import lightning.pytorch as pl
import torchmetrics


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
        mask_ratio = 0.75,
        norm_first = True,
        trunc_init = False,
        d_hid = 128,
        kernel_size = 3,
        stride = 1,
        padding =1,
        scale_mode = 'adaptive_scale',
        forecast_ratio=0.25,
        forecast_steps = 10,
        n_clusters = 0,
        dist_metric='cid',
        cls_embed=False,
        diagonal_attention=False,
        weights = None,
        z_type = 'vanila',
        bag_size= 1024,
        lr = 1e-3):

        super().__init__()
        self.model = MaskedAutoencoder(in_chans,seq_len,embed_dim,depth,num_heads,
                                       decoder_embed_dim,decoder_depth,decoder_num_heads,
                                       dropout,mask_ratio,norm_first,trunc_init,d_hid,kernel_size,stride,padding,
                                       scale_mode = scale_mode, 
                                       forecast_ratio=forecast_ratio,forecast_steps=forecast_steps,
                                       n_clusters = n_clusters,dist_metric=dist_metric,cls_embed=cls_embed,
                                       diagonal_attention=diagonal_attention,weights = weights,
                                       z_type = z_type,bag_size = bag_size)
        if cls_embed:
            self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=len(weights),average='macro')
            self.valid_f1 = torchmetrics.Accuracy(task="multiclass", num_classes=len(weights),average='macro')
        else:
            self.train_f1 = None
            self.valid_f1 = None
            
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            batch,y = batch
        else:
            y = None

        loss, logits, pred,cl_loss,cls_loss,space_loss = self.model(batch,y)
        loss_removed , loss_seen, forecast_loss, backcast_loss = loss

        loss = loss_removed + 0.75 * forecast_loss + 0.5 * loss_seen + 0.2 * backcast_loss
        if cl_loss:
            loss += cl_loss
            self.log("train/clustering_loss", cl_loss, sync_dist=True)

        if cls_loss:
            loss +=cls_loss
            batch_value = self.train_f1(logits, y)
            self.log("train/cls_loss", cls_loss, sync_dist=True)
            self.log('train/F1_step', batch_value)
        
        if space_loss:
            loss +=space_loss
            self.log("train/space_loss", space_loss, sync_dist=True)

        self.log("train/loss_removed", loss_removed, sync_dist=True)
        self.log("train/loss_seen", loss_seen, sync_dist=True)
        self.log("train/forecast_loss", forecast_loss, sync_dist=True)
        self.log("train/backcast_loss", backcast_loss, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self,):
        if self.valid_f1:
            self.train_f1.reset()
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            batch,y = batch
        else:
            y = None
        loss, logits, pred,cl_loss,cls_loss,space_loss = self.model(batch,y)
        loss_removed , loss_seen, forecast_loss, backcast_loss = loss
        loss = loss_removed + 0.75 * forecast_loss + 0.5 * loss_seen + 0.2 * backcast_loss
        
        if cl_loss:
            loss += cl_loss
            self.log("eval/clustering_loss", cl_loss, sync_dist=True)
        
        if cls_loss:
            loss +=cls_loss
            self.valid_f1.update(logits, y)
            self.log("eval/cls_loss", cls_loss, sync_dist=True)
        
        if space_loss:
            loss +=space_loss
            self.log("eval/space_loss", space_loss, sync_dist=True)


        self.log("eval/loss_removed", loss_removed, sync_dist=True)
        self.log("eval/loss_seen", loss_seen, sync_dist=True)
        self.log("eval/forecast_loss", forecast_loss, sync_dist=True)
        self.log("eval/backcast_loss", backcast_loss, sync_dist=True)
        self.log("eval/loss", loss, sync_dist=True)

        return loss
    
    def on_validation_epoch_end(self,):
        if self.valid_f1:
            self.log('eval/F1_step', self.valid_f1.compute())
            self.valid_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer



