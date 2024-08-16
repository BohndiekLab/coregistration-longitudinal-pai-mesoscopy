# -*- coding: utf-8 -*-
"""
Convert binary segmentations to distance-transformed images.

Not for clinical use.
SPDX-FileCopyrightText: 2024 Cancer Research UK Cambridge Institute, University of Cambridge, Cambridge, UK
SPDX-FileCopyrightText: 2024 Thierry L. Lefebvre
SPDX-FileCopyrightText: 2024 Sarah E. Bohndiek
SPDX-License-Identifier: MIT
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image, ImageSequence
from scipy import ndimage

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
outPath = 'PATH FOR SAVING DISTANCE IMAGES'

# List of filenames to process
namesnames = os.listdir(imgPath)

for namename in namesnames:
    # Load segmented image
    img = read_tiff(imgPath+namename)
    img[img!=0]=1 # Ensure segmentation is binary

    z_size = img.shape[0]
    image_stack = []

    # Convert segmentation to distance-transformed image
    for slice_idx in range(z_size):
        if np.sum(img[slice_idx,:,:])!=0:
            # Apply Exact Euclidean distance transform on non-empty slices
            img_out = ndimage.distance_transform_edt(img[slice_idx,:,:])
        else:
            img_out = np.zeros((600,600))
            
        # Build distance-transformed volume
        if slice_idx==0:
            image_stack = img_out
        elif slice_idx==1:
            image_stack = np.stack([image_stack,img_out],axis=0)
        else:
            image_stack = np.concatenate([image_stack,img_out[None]],axis=0)

    # Reorder dimensions for saving images as NIFTI
    image_stack = np.moveaxis(image_stack,[0,1,2],[2,1,0])
    
    # Export as NIFTI
    name_out = namename[:-5]+'.nii.gz'
    img_nii = nib.Nifti1Image(image_stack, np.eye(4))  # Save axis for data (just identity)
    img_nii.header.get_xyzt_units()
    img_nii.to_filename(os.path.join(outPath,name_out))  

    print(name_out+'    completed!')
