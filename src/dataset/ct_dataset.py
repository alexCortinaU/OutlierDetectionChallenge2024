from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import ast
from pathlib import Path
import pandas as pd
import torchio as tio

class CTDataset(Dataset):
    def __init__(self, data_path, split_ids, transform=None, labelling='binary'):

        if not Path(data_path).exists():
            raise FileNotFoundError(f'File {data_path} does not exist')
        
        csv_path = Path(data_path) / 'metadata.csv'
        if not Path(csv_path).exists():
            raise FileNotFoundError(f'File {csv_path} does not exist')
        
        self.data_path = Path(data_path)
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['sample_id'].isin(split_ids)]
        self.df = self.df.reset_index(drop=True)
        if labelling == 'binary':
            self.df['label'] = self.df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        else:
            self.df['label'] = self.df['label'].map({'normal': 0,
                                                     'warp': 1,
                                                     'sphere_water': 2,
                                                     'sphere_mean': 3,})
        print(f'Loaded {len(self.df)} samples')
        print(f'Label distribution: {self.df["label"].value_counts()}')

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        img_path = self.data_path / 'crops' / self.df['img_name'].iloc[idx]
        # img = nib.load(img_path).get_fdata()
        img = tio.ScalarImage(img_path)
            
        if self.transform:  
            img = self.transform(img)
        
        return {'image': img.data, 
                'lable': self.df['label'].iloc[idx]}

class CTDataModule(LightningDataModule):
    """
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_path: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        voxel_size=(1, 1, 1),
        crop_size=(128, 128, 128),
        train_ids=None,
        val_ids=None,
        labelling='binary',
        transforms = None
    ) -> None:
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        if transforms is None:
            self.transforms = tio.Compose([
                            tio.transforms.ToCanonical(),
                            tio.transforms.Resample(voxel_size),
                            tio.transforms.CropOrPad(crop_size, padding_mode='minimum'), # padding_mode=0), #
                            tio.transforms.Clamp(-1000, 1000),
                            tio.transforms.RescaleIntensity(out_min_max=(-1.0, 1.0), in_min_max=(-1000, 1000))
                            ])  
        else:
            self.transforms = transforms
        
        self.data_train = CTDataset(data_path=data_path,
                                    split_ids=train_ids,
                                    transform=self.transforms,
                                    labelling=labelling)
        self.data_val = CTDataset(data_path=data_path,
                                  split_ids=val_ids,
                                  transform=self.transforms,
                                  labelling=labelling)
        # self.data_val: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            drop_last=True,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )




if __name__ == "__main__":
    _ = CTDataModule()