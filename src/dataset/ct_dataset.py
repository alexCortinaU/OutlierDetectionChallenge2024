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

class CBCTDataset(Dataset):
    def __init__(self, data_path, train=True, crop_to_body=True, transform=None):

        if not Path(data_path).exists():
            raise FileNotFoundError(f'File {data_path} does not exist')
        
        csv_path = Path(data_path) / 'cbct_dataset.csv'
        if not Path(csv_path).exists():
            raise FileNotFoundError(f'File {csv_path} does not exist')
        
        self.data_path = Path(data_path)
        self.crop_to_body = crop_to_body
        self.transform = transform
        self.split = 'train' if train else 'test'
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df.split == self.split].reset_index(drop=True)
        self.df['margins_to_crop'] = self.df['margins_to_crop'].apply(ast.literal_eval)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        img_path = self.data_path / self.split / self.df['cbct_file'].iloc[idx]
        # img = nib.load(img_path).get_fdata()
        img = tio.ScalarImage(img_path)

        if self.crop_to_body:
            margins_to_crop = self.df['margins_to_crop'].iloc[idx]
            img = tio.Crop(margins_to_crop)(img)
            
        if self.transform:  
            img = self.transform(img)
        
        return img.data

class CBCTDataModule(LightningDataModule):
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
        patch_size = 16,
        pred_mask_scale = (0.2, 0.8),
        enc_mask_scale = (0.2, 0.8),
        aspect_ratio = (0.3, 3.0),
        num_enc_masks = 1,
        num_pred_masks = 4,
        block_depth = 3,
        allow_overlap = False,
        min_keep = 10,
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
        
        self.mask_collator = MaskCollator3D(
            input_size=crop_size,
            patch_size=patch_size,
            pred_mask_scale=pred_mask_scale,
            enc_mask_scale=enc_mask_scale,
            aspect_ratio=aspect_ratio,
            nenc=num_enc_masks,
            npred=num_pred_masks,
            block_depth=block_depth,
            allow_overlap=allow_overlap,
            min_keep=min_keep
        )

        self.data_train = CBCTDataset(data_path=data_path, train=True, crop_to_body=True, transform=self.transforms)
        self.data_val = CBCTDataset(data_path=data_path, train=False, crop_to_body=True, transform=self.transforms)
        # self.data_val: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    # @property
    # def num_classes(self) -> int:
    #     """Get the number of classes.

    #     :return: The number of MNIST classes (10).
    #     """
    #     return 10

    # def prepare_data(self) -> None:
    #     """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
    #     within a single process on CPU, so you can safely add your downloading logic within. In
    #     case of multi-node training, the execution of this hook depends upon
    #     `self.prepare_data_per_node()`.

    #     Do not use it to assign state (self.x = y).
    #     """
    #     MNIST(self.hparams.data_dir, train=True, download=True)
    #     MNIST(self.hparams.data_dir, train=False, download=True)

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
            collate_fn=self.mask_collator,
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
            collate_fn=self.mask_collator,
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

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = CBCTDataModule()