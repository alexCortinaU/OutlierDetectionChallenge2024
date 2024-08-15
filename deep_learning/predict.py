import torch
from torch import nn
from torch.utils.data import DataLoader
import torchio as tio
from pathlib import Path
import pandas as pd
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageFolder
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.sammed_class import FineTuneModule
from src.dataset.ct_dataset import CTDataModule, CTDatasetTEST
from tqdm import tqdm

from src.models.build_sam3D import sam_model_registry3D
from src.models.sammed_class import FineTuneModule
from src.models.components.attentive_pooler import AttentiveClassifier


this_path = Path().resolve()
data_path = Path("/media/7tb_encrypted/od_chall/dataset/challenge_data")
crops_path = data_path / "test/crops"

def main():
    # Load the pretrained model from checkpoint
    ckpt_path = data_path.parent / 'sam_med3d_turbo.pth'
    sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None) #vit_b_ori
    with open(ckpt_path, "rb") as f:
            state_dict = torch.load(f)
            sam_model.load_state_dict(state_dict['model_state_dict'])

    classifier = AttentiveClassifier(embed_dim=384,
                                    num_classes=1,
                                    num_heads=8,)

    checkpoint_path = "/home/alejandrocu/OutlierDetectionChallenge2024/deep_learning/SS24/us876b33/checkpoints/epoch=9-step=5810.ckpt"
    model = FineTuneModule.load_from_checkpoint(checkpoint_path,
                                                image_encoder=sam_model.image_encoder,
                                                classifier=classifier)

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
    test_ids = pd.read_csv(this_path.parent / "challenge_results/test_files_200.txt", header=None)[0].tolist()
    print(f'Number of test samples: {len(test_ids)}')
    df = pd.DataFrame({'sample_id': test_ids})

    data_test = CTDatasetTEST(data_path=data_path/'test',
                              test_df=df,
                              transform=transforms)
    dataloader = DataLoader(data_test, batch_size=1, shuffle=False)

    print(f'Loaded {len(data_test)} samples')
    print(f'{len(dataloader)} batches')

    # Run prediction
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    threshold = 0.5
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            images = batch["image"].to(device)
            output = model(images)
            logit = output.cpu().numpy()
            # print(np.squeeze(logit))
            bin_pred = 1 if logit >= threshold else 0
            # print(bin_pred)
            predictions.append({'scan_id': batch['id'][0],
                                'outlier': bin_pred,
                                # 'logit': np.squeeze(logit)
                                })

    import json
    # Write results to JSON file
    with open(this_path/'test_results_sammed_final.json', 'w') as json_file:
        json.dump(predictions, json_file, indent=4)
    # # Generate the final results file
    # results_file = "/path/to/results.txt"
    # with open(results_file, "w") as f:
    #     for pred in binarized_predictions:
    #         f.write(str(pred) + "\n")

if __name__ == "__main__":
    main()