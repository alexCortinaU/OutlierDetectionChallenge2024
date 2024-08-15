import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import MeanMetric, MinMetric

import torch.nn as nn
import torch.optim as optim

class FineTuneModule(pl.LightningModule):
    def __init__(self,
                 image_encoder: torch.nn.Module,
                 classifier: torch.nn.Module,
                 loss: torch.nn.BCELoss,
                 optimizer: torch.optim.AdamW,
                 scheduler: torch.optim.lr_scheduler,
                 num_epochs: int,
                 lr: float,
                 ):
        super().__init__()

        self.save_hyperparameters(logger=True,
                                  ignore=['image_encoder',
                                          'pooler',
                                          'head'])
        # freeze image encoder
        self.image_encoder = image_encoder
        self.image_encoder.eval()
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.classifier = classifier
        self.min_val_loss = MinMetric()
        self.val_loss = MeanMetric()
        self.val_r2 = R2Score()
        self.val_pearson = PearsonCorrCoef()
        self.val_step_outs = []
        self.loss = loss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            x = self.image_encoder(x, return_embds=True)

        x = self.pooler(x)
        if isinstance(self.pooler, AttentivePooler):
            x = x.squeeze(1)
        
        logits = self.head(x)
        loss = self.loss(logits, y.unsqueeze(1))

        self.log('train/loss',
                 loss, on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            x = self.image_encoder(x, return_embds=True)
            
        x = self.pooler(x)
        if isinstance(self.pooler, AttentivePooler):
            x = x.squeeze(1)
        logits = self.head(x)

        self.val_step_outs.append(torch.stack([logits.squeeze().clone().detach(),
                                               y.clone().detach()], dim=1))

        loss = self.loss(logits, y.unsqueeze(1))
        self.val_loss.update(loss)

        r2score = self.val_r2(logits.squeeze(), y.squeeze())
        pcorr = self.val_pearson(logits.squeeze(), y.squeeze())

        self.log('val/loss',
                 loss, on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        
        self.log('val/r2',
                 r2score, on_step=False,
                 on_epoch=True,
                 sync_dist=True)
        
        self.log('val/pearson',
                 pcorr, on_step=False,
                 on_epoch=True,
                 sync_dist=True)
        
        return loss
    def __init__(self):
        super().__init__()
        # Define your image_encoder and AttentiveClassifier here
        
    def forward(self, x):
        # Implement the forward pass of your model here
        pass
    
    def training_step(self, batch, batch_idx):
        # Implement the training step logic here
        pass
    
    def validation_step(self, batch, batch_idx):
        # Implement the validation step logic here
        pass
    
    def test_step(self, batch, batch_idx):
        # Implement the test step logic here
        pass
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer