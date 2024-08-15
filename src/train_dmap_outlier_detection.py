import vtk
import numpy as np
import os
from pathlib import Path
import argparse
from dtu_spine_config import DTUConfig
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import nibabel as nib

def compute_template_analysis(settings):
    print("Running PCA analysis")
    data_dir = settings["data_dir"]
    dist_fields_dir = os.path.join(data_dir, "train", "dist_fields")
    training_list = settings["data_set"]
    result_dir = settings["result_dir"]

    pca_dir = os.path.join(result_dir, "temp_dist_fields_analysis")
    
    # Create folders if they don't exist
    Path(pca_dir).mkdir(parents=True, exist_ok=True)
    # pca_analysis_out = os.path.join(pca_dir, f"pca_analysis.pkl")
    mean_shape_name = os.path.join(pca_dir, f"mean_dist_fields.nii.gz")

    training_id_list_file = os.path.join(result_dir, training_list)
    all_scan_ids = np.loadtxt(str(training_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} samples in {training_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return
    
    # Read the first distance field to determine the shape
    id_0 = all_scan_ids[0].strip()
    dist_field_name  = os.path.join(dist_fields_dir, f"{id_0}_dist_field_crop.nii.gz")
    dist_field_proxy = nib.load(dist_field_name)
    dist_field = dist_field_proxy.get_fdata()
    shape = dist_field.shape
    nvoxels = np.prod(shape)
    
    n_samples = len(all_scan_ids)
    print("Creating data matrix of size {} x {}".format(n_samples, nvoxels))
    data_matrix = np.zeros((n_samples, nvoxels))
    
    # Now read all distance fields
    i = 0
    for idx in all_scan_ids:
        print("Reading {}/{}".format(i + 1, n_samples))
        scan_id = idx.strip()
        dist_field_name = os.path.join(dist_fields_dir, f"{scan_id}_dist_field_crop.nii.gz")
        dist_field_proxy = nib.load(dist_field_name)
        dist_field = dist_field_proxy.get_fdata()
        if dist_field.shape != shape:
            print(f"Shape of {scan_id} is {dist_field.shape} and it should be {shape}")
            return
        data_matrix[i, :] = dist_field.flatten()
        i += 1
        
    average_shape = np.mean(data_matrix, 0)
    reshaped_average_shape = average_shape.reshape(shape)
    
    # Save the mean shape
    
    # n_components = 10
    # print(f"Computing PCA with {n_components} components")
    # shape_pca = PCA(n_components=n_components)
    # shape_pca.fit(data_matrix)
    # components = shape_pca.transform(data_matrix)    
    
    # print(f"Saving {pca_analysis_out}")
    # with open(pca_analysis_out, 'wb') as pickle_file:
    #     pickle.dump(shape_pca, pickle_file)

    # plt.plot(shape_pca.explained_variance_ratio_ * 100)
    # plt.xlabel('Principal component')
    # plt.ylabel('Percent explained variance')
    # plt.show()
    
    # n_modes = 5
    # print(f"Synthesizing shapes using {n_modes} modes")
    # for i in range(n_modes):
    #     synthesized_shape = reshaped_average_shape + shape_pca.components_[i, :].reshape(shape)
    #     synthesized_shape_name = os.path.join(pca_dir, f"synthesized_shape_mode_{i}.nii.gz")
    #     synthesized_shape_proxy = nib.Nifti1Image(synthesized_shape, dist_field_proxy.affine)
    #     nib.save(synthesized_shape_proxy, synthesized_shape_name)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='train-pdm_outlier_detection')
    config = DTUConfig(args)
    print(args)
    if config.settings is not None:
        compute_template_analysis(config.settings)