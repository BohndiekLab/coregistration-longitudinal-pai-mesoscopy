# -*- coding: utf-8 -*-
"""
Point-to-plane iterative closest point (ICP) algorithm for the co-registration of mesoscopic photoacoustic images.
Fixed and moving segmented images are converted to point clouds and an affine transformation matrix is calculated for the co-registration of point clouds using ICP.

Not for clinical use.
SPDX-FileCopyrightText: 2024 Cancer Research UK Cambridge Institute, University of Cambridge, Cambridge, UK
SPDX-FileCopyrightText: 2024 Thierry L. Lefebvre
SPDX-FileCopyrightText: 2024 Sarah E. Bohndiek
SPDX-License-Identifier: MIT
"""
import os
import glob
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
from PIL import Image
import scipy.ndimage

def read_tiff(path):
    """
    Reads a multipage TIFF file and returns it as a NumPy array.
    
    Parameters:
    path (str): Path to the multipage-TIFF file.
    
    Returns:
    np.array: A NumPy array containing the image stack.
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)


# Paths to input images and output results
imgPath = 'PATH TO SEGMENTED IMAGES'
outPath = 'PATH FOR SAVING AFFINE TRANSFORMS'

# List of filenames and IDs to process
namesnames = os.listdir(imgPath)
IDList = np.array(['INSERT LIST OF IDENTIFIERS GROUPING IMAGES TO CO-REGISTER']) 

for ID in IDList:
    ii = 0
    print(ID)
    namesID = [matchin for matchin in namesnames if ID in matchin]
    print(namesID)

    # Load fixed image (reference image for co-registration)
    fixed_name = namesID[0]
    print(fixed_name)
    fixed_image = read_tiff(imgPath + fixed_name)
    fixed_image[fixed_image != 0] = 1 # Ensure segmentation is binary

    # Remove small objects from the binary image
    fixed_image = ski.morphology.remove_small_objects(ski.measure.label(fixed_image, background=0), min_size=175, connectivity=26)

    # Convert segmented image to point clouds for ICP
    verts1, faces1, normals1, values1 = ski.measure.marching_cubes(fixed_image, 0.0)
    pcd_fixed = o3d.geometry.PointCloud()
    pcd_fixed.points = o3d.utility.Vector3dVector(verts1)
    pcd_fixed.normals = o3d.utility.Vector3dVector(normals1)    

    namesID = namesID[1:] # All other images are the moving images

    # Conduct iterative co-registration of moving images to fixed image
    for moving_name in namesID:

        print('----------------------------')
        print(fixed_name)
        print('co-registered with')
        print(moving_name)
        print('----------------------------')
        
        # Load moving image (to be co-registered to the fixed image)
        moving_image = read_tiff(imgPath + moving_name)
        moving_image[moving_image != 0] = 1 # Ensure segmentation is binary
        
        # Remove small objects from the binary image
        moving_image = ski.morphology.remove_small_objects(ski.measure.label(moving_image, background=0), min_size=175, connectivity=26)
        
        # Convert segmented image to point clouds for ICP
        verts2, faces2, normals2, values2 = ski.measure.marching_cubes(moving_image, 0.0)
        pcd_moving = o3d.geometry.PointCloud()
        pcd_moving.points = o3d.utility.Vector3dVector(verts2)
        pcd_moving.normals = o3d.utility.Vector3dVector(normals2)

        # Apply Point-to-Plane ICP for registration
        print("Apply point-to-plane ICP")
        trans_init = np.eye(4)  # Initial transformation (identity matrix)
        threshold = 250  # Distance threshold for ICP
        
        icp_transf = o3d.pipelines.registration.registration_icp(
            pcd_fixed, pcd_moving, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
        
        # Apply the transformation to the moving segmented image
        moved_image = scipy.ndimage.affine_transform(moving_image, icp_transf.transformation)

        # Save the affine transformation matrix to an Excel file
        df = pd.DataFrame(icp_transf.transformation)
        df.to_excel(outPath + moving_name[0:-5] + '.xlsx')
        ii += 1
