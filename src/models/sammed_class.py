import torch
import lightning as pl
from torchmetrics import MeanMetric, MinMetric, Accuracy

import torch.nn as nn
import torch.optim as optim

class FineTuneModule(pl.LightningModule):
    def __init__(self,
                 image_encoder: torch.nn.Module,
                 classifier: torch.nn.Module,
                 lr: float,
                 loss = torch.nn.BCEWithLogitsLoss, #torch.nn.BCELoss, #torch.nn.CrossEntropyLoss,
                 optimizer = torch.optim.AdamW,
                #  scheduler: torch.optim.lr_scheduler,
                 num_epochs: int = 10,
                 ):
        super().__init__()

        self.save_hyperparameters(logger=True,
                                  ignore=['image_encoder',
                                          'classifier'])
        # freeze image encoder
        self.image_encoder = image_encoder
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.classifier = classifier
        self.min_val_loss = MinMetric()
        self.val_loss = MeanMetric()
        self.val_acc = Accuracy(task='binary')
        self.train_acc = Accuracy(task='binary')
        self.loss = loss()
    
    def forward(self, x):
        if isinstance(x, dict):
            x = x['image']
        x = self.image_encoder(x)
        B, L = x.shape[:2]
        x = x.reshape(B, L, -1).permute(0, 2, 1)
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        
        with torch.no_grad():
            x = self.image_encoder(x)

        B, L = x.shape[:2]
        x = x.reshape(B, L, -1).permute(0, 2, 1)

        logits = self.classifier(x)
        loss = self.loss(logits, y.unsqueeze(1).float())
        acc = self.train_acc(logits.squeeze(), y.squeeze())

        self.log('train/loss',
                 loss, on_step=True,
                 on_epoch=True,
                 sync_dist=True, 
                 prog_bar=True)
        self.log('train/acc',
                 acc, on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']

        with torch.no_grad():
            x = self.image_encoder(x)
        
        B, L = x.shape[:2]
        x = x.reshape(B, L, -1).permute(0, 2, 1)

        logits = self.classifier(x)
        loss = self.loss(logits, y.unsqueeze(1).float())
        self.val_loss.update(loss)

        acc = self.val_acc(logits.squeeze(), y.squeeze())

        self.log('val/loss',
                 loss, on_step=True,
                 on_epoch=True,
                 sync_dist=True,
                 prog_bar=True)
        
        self.log('val/acc',
                 acc, on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 prog_bar=True)
        
        return loss
    
    def on_train_start(self):
        self.val_loss.reset()
    
    def configure_optimizers(self):
        param_groups = [
            {'params': self.classifier.parameters()},
        ]
        optimizer = self.hparams.optimizer(params=param_groups,
                                           lr=self.hparams.lr,
                                           eps=1e-8)
        # if self.hparams.scheduler:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': scheduler,
        #             'monitor': 'val_loss',
        #             'interval': 'epoch',
        #             'frequency': 1,
        #         }
        #     }
        
        return {'optimizer': optimizer}