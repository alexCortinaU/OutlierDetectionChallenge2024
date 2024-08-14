import nibabel as nib
#from radiomics import featureextractor
import os
import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

#load nii files from data folder
def load_nii(file_path):
    img = sitk.ReadImage(file_path)
    return img

dataFolder = "/mrhome/kristine/Documents/02985_SummerSchool/challenge_data/train/crops/"
pattern = os.path.join(dataFolder, '*crop.nii.gz')
mask_pattern_1 = os.path.join(dataFolder, '*sphere_outlier_water.nii.gz')
mask_pattern_2 = os.path.join(dataFolder, '*sphere_outlier_mean_std_inpaint.nii.gz')
mask_pattern_3 = os.path.join(dataFolder, '*label_warp_outlier.nii.gz')

# Get a list of all files that match the pattern
nii_mask_files_1 = glob.glob(mask_pattern_1)
nii_mask_files_2 = glob.glob(mask_pattern_2)
nii_mask_files_3 = glob.glob(mask_pattern_3)

nii_raw_files = glob.glob(pattern)

# Load each NIfTI file using nibabel and store in a list
img_mask_1_all = []
img_mask_2_all = []
img_mask_3_all = []
img_raw_all = []

for file in nii_mask_files_1[1:100]:
    img_mask_obj_1 = load_nii(file)
    img_mask_1 = sitk.GetArrayFromImage(img_mask_obj_1)
    img_mask_1_all.append(img_mask_1)

for file in nii_mask_files_2[1:100]:
    img_mask_obj_2 = load_nii(file)
    img_mask_2 = sitk.GetArrayFromImage(img_mask_obj_2)
    img_mask_2_all.append(img_mask_2)

for file in nii_mask_files_3[1:100]:
    img_mask_obj_3 = load_nii(file)
    img_mask_3 = sitk.GetArrayFromImage(img_mask_obj_3)
    img_mask_3_all.append(img_mask_3)

for file in nii_raw_files[1:100]:
    img_raw_obj = load_nii(file)
    img_raw = sitk.GetArrayFromImage(img_raw_obj)
    img_raw_all.append(img_raw)

# Calculate the volume of the mask for each image
volume_1_all = []
volume_2_all = []
volume_3_all = []
volume_raw_all = []

for i, img in enumerate(img_mask_1_all):
    volume_1 = img.sum()
    volume_1_all.append(volume_1)
    print(f"Volume of mask {i+1}: {volume_1}")

for i, img in enumerate(img_mask_2_all):
    volume_2 = img.sum()
    volume_2_all.append(volume_2)
    print(f"Volume of mask {i+1}: {volume_2}")

for i, img in enumerate(img_mask_3_all):
    volume_3 = img.sum()
    volume_3_all.append(volume_3)
    print(f"Volume of mask {i+1}: {volume_3}")

for i, img in enumerate(img_raw_all):
    volume_raw = img.sum()
    volume_raw_all.append(volume_raw)
    print(f"Volume of mask {i+1}: {volume_raw}")



plt.figure()
plt.hist(volume_2_all, bins=100)
plt.title('Histogram of mask volumes')
plt.xlabel('Volume 2')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.hist(volume_1_all, bins=100)
plt.title('Histogram of mask volumes')
plt.xlabel('Volume')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.hist(volume_raw_all, bins=100)
plt.title('Histogram of mask volumes')
plt.xlabel('Volume')
plt.ylabel('Count')
plt.show()

#make boxplot comparing the volumes of the masks and the raw data

plt.figure()
plt.boxplot([volume_1_all, volume_2_all, volume_3_all, volume_raw_all])
plt.title('Boxplot of mask volumes')
plt.xlabel('Mask')
plt.ylabel('Volume')
plt.xticks([1, 2, 3, 4], ['Mask 1', 'Mask 2', 'Mask 3', 'Raw data'])
plt.show()


# Extract region properties
props = measure.regionprops(img_mask_1)

for prop in props:
    volume = prop.area  # Volume is equivalent to the number of voxels
    centroid = prop.centroid
    bbox = prop.bbox






