#%%
import vtk
import numpy as np
# Path to your .vtk file
#%%
file_path = "/Users/aoi/Downloads/challenge_data/train/surfaces/sample_1103_surface_sphere_outlier_water.vtk"

import os

import vtk

import numpy as np

import matplotlib.pyplot as plt

 

def vtk_to_vector(pd):

    n_points = pd.GetNumberOfPoints()

    vec = np.zeros(n_points * 3)

    for i in range(n_points):

        p = pd.GetPoint(i)

        vec[i*3] = p[0]

        vec[i*3+1] = p[1]

        vec[i*3+2] = p[2]

    return vec

 

def read_and_extract_coordinates(file_path):

    # Read the .vtk file

    reader = vtk.vtkPolyDataReader()

    reader.SetFileName(file_path)

    reader.Update()

 

    # Get the polydata object

    polydata = reader.GetOutput()

 

    # Convert the polydata to a NumPy vector

    vector = vtk_to_vector(polydata)

 

    # Extract x, y, and z coordinates separately

    x_coords = vector[0::3]  # Start at index 0, take every 3rd element

    y_coords = vector[1::3]  # Start at index 1, take every 3rd element

    z_coords = vector[2::3]  # Start at index 2, take every 3rd element

    return x_coords, y_coords, z_coords
# %%
a,b,c = read_and_extract_coordinates(file_path)

# %%
