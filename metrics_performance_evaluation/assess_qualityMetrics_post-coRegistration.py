# -*- coding: utf-8 -*-
"""
Assess quality metrics for evaluating co-registration performance.
Load the previously calculated spatial transformation and apply on both moving intensity image and segmentation to compute selected metrics.

Not for clinical use.
SPDX-FileCopyrightText: 2024 Cancer Research UK Cambridge Institute, University of Cambridge, Cambridge, UK
SPDX-FileCopyrightText: 2024 Thierry L. Lefebvre
SPDX-FileCopyrightText: 2024 Sarah E. Bohndiek
SPDX-License-Identifier: MIT
"""

import os
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
from skimage.metrics import structural_similarity, normalized_mutual_information
import matplotlib.pyplot as plt

def dice_coef(fixed_seg, moving_seg):
    """
    Calculate the Dice coefficient between two binary masks.
    """
    fixed_seg_f = fixed_seg.flatten()
    moving_seg_f = moving_seg.flatten()
    intersect_f = np.sum(fixed_seg_f * moving_seg_f)
    return (2. * intersect_f) / (np.sum(fixed_seg_f) + np.sum(moving_seg_f))

def evaluate_registration(fixed_img, warped_img, fixed_seg, warped_seg):
    """
    Evaluate the registration quality of the registered image and segmentation with respect to a template.
    """    
    # Compute structural similarity index (SSIM)
    ssim_value = structural_similarity(fixed_img, warped_img, data_range=warped_img.max() - warped_img.min())    

    # Compute normalized mutual information (NMI)
    nmi = normalized_mutual_information(fixed_img, warped_img)
    
    # Compute Dice coefficient
    dice = dice_coef(fixed_seg, warped_seg)

    # Convert segmentations to SimpleITK images
    fixed_seg_sitk = sitk.GetImageFromArray(fixed_seg)
    warped_seg_sitk = sitk.GetImageFromArray(warped_seg)
    z_size = fixed_seg_sitk.GetSize()[2]
    
    haus_list, mean_surface_list, median_surface_list, std_surface_list, max_surface_list = [], [], [], [], []

    for slice_idx in range(z_size):
        if np.sum(fixed_seg[slice_idx, :, :].flatten()) != 0 and np.sum(warped_seg[slice_idx, :, :].flatten()) != 0:
            reg_seg = warped_seg_sitk[:, :, slice_idx]
            temp_seg = fixed_seg_sitk[:, :, slice_idx]
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(temp_seg, reg_seg)
            haus_slice_value = hausdorff_distance_filter.GetHausdorffDistance()
            haus_list.append(haus_slice_value)
            
            # Symmetric surface distance measures
            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(temp_seg, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(temp_seg)
            registered_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reg_seg, squaredDistance=False, useImageSpacing=True))
            registered_surface = sitk.LabelContour(reg_seg)

            reg2ref_distance_map = reference_distance_map * sitk.Cast(registered_surface, sitk.sitkFloat32)
            ref2reg_distance_map = registered_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            reg2ref_distances = sitk.GetArrayViewFromImage(reg2ref_distance_map)[sitk.GetArrayViewFromImage(reg2ref_distance_map) != 0]
            ref2reg_distances = sitk.GetArrayViewFromImage(ref2reg_distance_map)[sitk.GetArrayViewFromImage(ref2reg_distance_map) != 0]

            all_surface_distances = np.concatenate([reg2ref_distances, ref2reg_distances])

            mean_surface_list.append(np.mean(all_surface_distances))

    
    mean_haus = np.mean(haus_list)
    mean_surface = np.mean(mean_surface_list)

    return ssim_value, nmi, dice, mean_haus, mean_surface

def read_tiff(path):
    """
    Reads a multipage TIFF file and returns it as a NumPy array.
    
    Parameters:
    path (str): Path to the multipage-TIFF file.
    
    Returns:
    np.array: A NumPy array containing the image stack.
    """
    img = Image.open(path)
    images = [np.array(img.seek(i) or img) for i in range(img.n_frames)]
    return np.array(images)




# Paths to input intensity images, segmentations, and transforms, and to output evaluated quality metrics
imgPath = 'PATH TO INTENSITY IMAGES'
segPath = 'PATH TO SEGMENTED IMAGES'
tfmPath = 'PATH TO SAVED TRANSFORMS'
outPath = 'PATH FOR SAVING METRICS'

# List of filenames and IDs to process
namesnames = os.listdir(imgPath)
IDList = np.array(['INSERT LIST OF IDENTIFIERS GROUPING IMAGES TO CO-REGISTER'])

landm_master = pd.read_excel('LOAD SAVED MANUAL LANDMARKS')

# Prepare lists to store the evaluation metrics
ssimList, nmiList, diceList, haus_meanList, mean_surfaceList, tre, nameListout, name3Listout = [], [], [], [], [], [], [], []

spacing = [1, 1, 1]

for ID in IDList:
    ii = 0
    print(ID)
    namesID = [matchin for matchin in namesnames if ID in matchin]
    print(namesID)

    # Load fixed intensity image 
    fixed_name = namesID[0]
    fixed_image = read_tiff(imgPath + fixed_name)
    fixed_image_sitk = sitk.GetImageFromArray(fixed_image)
    fixed_image_sitk.SetSpacing(spacing)

    # Find the corresponding 3D coordinates of landmarks in the fixed image 
    landm_df = landm_master.loc[landm_master["Animal ID"] == fixed_name]
    landm_fixed = landm_df.iloc[0]

    # Load fixed segmented image 
    fixed_image_seg = read_tiff(segPath + fixed_name)
    fixed_image_seg_sitk = sitk.GetImageFromArray(fixed_image_seg)
    fixed_image_seg_sitk.SetSpacing(spacing)

    namesID = namesID[1:] # All other images are the moving images

    # Apply iteratively the transform to moving image to assess quality metrics
    for moving_name in namesID:
        
        # Load moving intensity image 
        moving_image = read_tiff(imgPath + moving_name)
        moving_image_sitk = sitk.GetImageFromArray(moving_image)
        moving_image_sitk.SetSpacing(spacing)
        
        # Load moving segmented image 
        moving_image_seg = read_tiff(segPath + moving_name)
        moving_image_seg_sitk = sitk.GetImageFromArray(moving_image_seg)
        moving_image_seg_sitk.SetSpacing(spacing)

        # Find the corresponding 3D coordinates of landmarks in the moving image 
        landm_df = landm_master.loc[landm_master["Animal ID"] == moving_name]
        landm_moving = landm_df.iloc[1]

        # Read the previously calculated spatial transformation
        transf_name, _ = os.path.splitext(moving_name)
        final_transform = sitk.ReadTransform(tfmPath + transf_name + ".tfm") # Adjust the loading of calculated transform if non-SimpleITK-based method used

        # Apply the transform to the moving image and segmentation
        warped_image_sitk = sitk.Resample(moving_image_sitk, fixed_image_sitk, final_transform, sitk.sitkLinear, 0.0, moving_image_sitk.GetPixelID())
        warped_image_seg_sitk = sitk.Resample(moving_image_seg_sitk, fixed_image_seg_sitk, final_transform, sitk.sitkNearestNeighbor, 0.0, moving_image_seg_sitk.GetPixelID())

        # Convert co-registered images back to NumPy arrays
        warped_image = sitk.GetArrayFromImage(warped_image_sitk)
        warped_image_seg = sitk.GetArrayFromImage(warped_image_seg_sitk)

        # Evaluate registration metrics
        ssim_value, nmi, dice, mean_haus, mean_surface = evaluate_registration(
            fixed_image, warped_image, fixed_image_seg, warped_image_seg
        )

        # Store the quality metrics for saving
        ssimList.append(ssim_value)
        nmiList.append(nmi)
        diceList.append(dice)
        haus_meanList.append(mean_haus)
        mean_surfaceList.append(mean_surface)
        nameListout.append(moving_name)

        # Calculate target registration errors (TRE) - format specific to loaded spatial coordinates in 3D saved in an Excel file (3 landmarks per image)
        fixed_landmarks = [(int(landm_fixed[f"MatchZ{i}"]), int(landm_fixed[f"MatchX{i}"]), int(landm_fixed[f"MatchY{i}"])) for i in range(1, 4)]
        moving_landmarks = [(int(landm_moving[f"MatchZ{i}"]), int(landm_moving[f"MatchX{i}"]), int(landm_moving[f"MatchY{i}"])) for i in range(1, 4)]

        for fixed_point, moving_point in zip(fixed_landmarks, moving_landmarks):
            fixed_point_transformed = final_transform.TransformPoint(fixed_point)
            distance = np.linalg.norm(np.array(fixed_point_transformed) - np.array(moving_point))
            tre.append(distance)
            name3Listout.append(moving_name)

# Save the calculated quality metrics
dataout = {
    'Name': nameListout,
    'SSIM': ssimList,
    'NMI': nmiList,
    'Dice': diceList,
    'Haus_Mean': haus_meanList,
    'Mean_Distance': mean_surfaceList
}
dfout = pd.DataFrame(dataout)
dfout.to_excel(outPath+'NAME OF FILE TO SAVE.xlsx')

# Save TRE seperately since more than one value is calculated by co-registration pair 
dataTRE_out = {
    'Name': name3Listout,
    'TRE':   tre,
} # Here was saved 3 TRE per pair
dfTRE_out = pd.DataFrame(dataTRE_out)
dfTRE_out.to_excel(outPath+'NAME OF TRE FILE TO SAVE.xlsx')