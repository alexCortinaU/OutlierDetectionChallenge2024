import pandas as pd
import SimpleITK as sitk
import six
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from radiomics import featureextractor  # Assuming this is the correct import for your feature extractor
data_path = Path("/media/7tb_encrypted/od_chall/dataset/challenge_data")
crops_path = data_path / "train/crops"
this_path = Path().resolve()
import glob, os

# Assuming this_path and crops_path are defined somewhere above this code
params = this_path / 'notebooks/Params.yaml'

file_types = {'normal': '',
              'sphere_mean': '_sphere_outlier_mean_std_inpaint',
              'sphere_water': '_sphere_outlier_water',
              'warp': '_warp_outlier'}

def process_sample(sample_file_type):
    sample, file_type = sample_file_type
    img_path = crops_path / f"{sample}_crop{file_type}.nii.gz"
    label_path = crops_path / f"{sample}_crop_label{file_type}.nii.gz"
    if not img_path.exists() or not label_path.exists():
        return None  # Or handle missing files as needed
    img = sitk.ReadImage(str(img_path))
    mask = sitk.ReadImage(str(label_path))

    extractor = featureextractor.RadiomicsFeatureExtractor(str(params))
    result = extractor.execute(img, mask)
    features = {'sample_id': sample, 'img_name': img_path.name, 'label_name': label_path.name, 'label': file_type}
    for key, value in six.iteritems(result):
        if key.startswith(('original_shape', 'original_firstorder', 'original_glcm')):
            features[key] = value
    return features

if __name__ == '__main__':

    train_ids = glob.glob(str(crops_path / "*_crop.nii.gz"))
    train_ids = [Path(x).name.split('_crop')[0] for x in train_ids]
    # Prepare data for multiprocessing
    sample_file_types = [(sample, file_type) for sample in train_ids for file_type in file_types.values()]
    
    # Initialize multiprocessing Pool
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()//2)
    
    # Process data in parallel with progress tracking
    results = []
    for result in tqdm(pool.imap_unordered(process_sample, sample_file_types), total=len(sample_file_types)):
        results.append(result)
    
    pool.close()
    pool.join()
    
    # Filter out None results if any files were missing
    results = [result for result in results if result is not None]
    
    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv('radiomics_features.csv', index=False)