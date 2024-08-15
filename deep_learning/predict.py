import torch
from torch import nn
from torch.utils.data import DataLoader
import torchio as tio
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.sammed_class import FineTuneModule
from src.dataset.ct_dataset import CTDataModule, CTDataset


this_path = Path().resolve()
data_path = Path("/media/7tb_encrypted/od_chall/dataset/challenge_data")
crops_path = data_path / "test/crops"

def main():
    # Load the pretrained model from checkpoint
    checkpoint_path = "/path/to/checkpoint.pt"
    model = FineTuneModule.load_from_checkpoint(checkpoint_path)

    # Create the datamodule

    voxel_size = (1, 1, 1) #(0.5, 0.5, 0.5) #(1.171875, 1.171875, 2.5) #(1.5, 1.5, 1.5)
    crop_size = (128, 128, 128) #(336, 224, 64) #(128, 128, 128)
    transforms = tio.Compose([
                            tio.transforms.ToCanonical(),
                            tio.transforms.Resample(voxel_size), # , image_interpolation='cosine'
                            tio.transforms.CropOrPad(crop_size, padding_mode=-1000), # padding_mode=0), #
                            tio.transforms.Clamp(-1000, 1000),
                            tio.transforms.RescaleIntensity(out_min_max=(0.0, 1.0), in_min_max=(-1000, 1000))
                            ])  
    dataset = ImageFolder(data_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Run prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy())

    # Binarize the predictions
    threshold = 0.5
    binarized_predictions = [1 if pred >= threshold else 0 for pred in predictions]

    # Generate the final results file
    results_file = "/path/to/results.txt"
    with open(results_file, "w") as f:
        for pred in binarized_predictions:
            f.write(str(pred) + "\n")

if __name__ == "__main__":
    main()