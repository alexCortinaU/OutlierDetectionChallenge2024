import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import sys, shutil
import wandb
wandb.login()

import pandas as pd
import torchio as tio
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split

this_path = Path().resolve()
data_path = Path("/media/7tb_encrypted/od_chall/dataset/challenge_data")
crops_path = data_path / "test/crops"



from src.models.build_sam3D import sam_model_registry3D
from src.dataset.ct_dataset import CTDataModule
from src.models.sammed_class import FineTuneModule
from src.models.components.attentive_pooler import AttentiveClassifier



# Initialize the Lightning DataModule
batch_size = 10
num_workers = 4
voxel_size = (1, 1, 1) #(0.5, 0.5, 0.5) #(1.171875, 1.171875, 2.5) #(1.5, 1.5, 1.5)
crop_size = (128, 128, 128) #(336, 224, 64) #(128, 128, 128)
lr = 1e-3
max_epochs = 20
# train_ids = pd.read_csv(this_path / "custom_train_list_100.txt", header=None)[0].tolist()
# val_ids = pd.read_csv(this_path / "custom_validation_list_100.txt", header=None)[0].tolist()
# sample_ids = glob.glob(str(crops_path / "*_crop.nii.gz"))
# sample_ids = [Path(x).name.split('_crop')[0] for x in sample_ids]
metadata = pd.read_csv(data_path / "train/metadata.csv")
sample_ids = metadata['sample_id'].unique()
train_ids, val_ids = train_test_split(sample_ids, test_size=0.2, random_state=42)

transforms = tio.Compose([
                            tio.transforms.ToCanonical(),
                            tio.transforms.Resample(voxel_size), # , image_interpolation='cosine'
                            tio.transforms.CropOrPad(crop_size, padding_mode=-1000), # padding_mode=0), #
                            tio.transforms.Clamp(-1000, 1000),
                            tio.transforms.RescaleIntensity(out_min_max=(0.0, 1.0), in_min_max=(-1000, 1000))
                            ])  

data_module = CTDataModule(data_path=data_path/'train', 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            voxel_size=voxel_size, 
                            crop_size=crop_size,
                            train_ids=train_ids,
                            val_ids=val_ids,
                            transforms=transforms)

# Initialize the Lightning Module
ckpt_path = data_path.parent / 'sam_med3d_turbo.pth'
sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None) #vit_b_ori
with open(ckpt_path, "rb") as f:
        state_dict = torch.load(f)
        sam_model.load_state_dict(state_dict['model_state_dict'])

classifier = AttentiveClassifier(embed_dim=384,
                                 num_classes=1,
                                 num_heads=8,)

model = FineTuneModule(image_encoder=sam_model.image_encoder,
                       classifier=classifier,
                       lr=lr)
# Initialize the Lightning Trainer
run_name = sys.argv[1]
ckp_path = this_path / f"SS24/{run_name}/checkpoints/"
if not ckp_path.exists:
    ckp_path.mkdir(parents=True)
wandb_logger = WandbLogger(log_model="all",
                           project="SS24",
                           name=run_name)
checkpoint_callback = ModelCheckpoint(monitor="val/loss",
                                      mode="min",
                                      dirpath=ckp_path,
                                      filename='{epoch}-{val/loss:.2f}')
trainer = pl.Trainer(devices=[0],
                     max_epochs=max_epochs,
                     logger=wandb_logger,)
# Train the model
trainer.fit(model, data_module)