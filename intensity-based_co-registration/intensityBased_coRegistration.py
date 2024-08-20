# -*- coding: utf-8 -*-
"""
Intensity-based co-registration of mesoscopic photoacoustic images.
An affine spatial transformation is iteratively calculated to align the moving to the fixed  images using gradient-descent-based minimisation of an optimiser metric (either mutual information [MI] or normalised cross-correlation [NCC]). This script is also used for aligning distance-transformed segmentations using NCC.

Not for clinical use.
SPDX-FileCopyrightText: 2024 Cancer Research UK Cambridge Institute, University of Cambridge, Cambridge, UK
SPDX-FileCopyrightText: 2024 Thierry L. Lefebvre
SPDX-FileCopyrightText: 2024 Sarah E. Bohndiek
SPDX-License-Identifier: MIT
"""

import os
import numpy as np
import SimpleITK as sitk
from PIL import Image


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
imgPath = 'PATH TO PREPROCESSED INTENSITY IMAGES (OR DISTANCED-TRANSFORMED IMAGES)'
outPath = 'PATH FOR SAVING AFFINE TRANSFORMS'

# List of filenames and IDs to process
namesnames = os.listdir(imgPath)
IDList = np.array(['INSERT LIST OF IDENTIFIERS GROUPING IMAGES TO CO-REGISTER']) 

# Initialise isotropic spacing for images
spacing = [1, 1, 1]

for ID in IDList:
    ii = 0
    print(ID)
    namesID = [matchin for matchin in namesnames if ID in matchin]
    print(namesID)
    
    # Load fixed image (reference image for co-registration)
    fixed_name = namesID[0]
    print(fixed_name)
    fixed_image = read_tiff(imgPath + fixed_name)

    # Convert the fixed image to SimpleITK format
    fixed_image_sitk = sitk.GetImageFromArray(fixed_image)
    fixed_image_sitk.SetOrigin((0, 0, 0))
    fixed_image_sitk.SetSpacing(spacing)

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

        # Convert the moving image to SimpleITK format
        moving_image_sitk = sitk.GetImageFromArray(moving_image)
        moving_image_sitk.SetOrigin((0, 0, 0))
        moving_image_sitk.SetSpacing(spacing)

        # Initialize the transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image_sitk, 
            moving_image_sitk, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )   
        
        # Set up the registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Similarity metric - using NCC (uncomment line below for MI) 
        registration_method.SetMetricAsCorrelation()
        #registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=45)        
        registration_method.SetMetricSamplingStrategy(registration_method.NONE)

        # Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer - Gradient Descent
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, 
            numberOfIterations=100, 
            convergenceMinimumValue=1e-6, 
            convergenceWindowSize=10
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Set initial transform and run registration
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = registration_method.Execute(
            sitk.Cast(fixed_image_sitk, sitk.sitkFloat32), 
            sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
        )

        # Resample the moving image and segmentation to align with the fixed image
        moved_image_sitk = sitk.Resample(
            moving_image_sitk, fixed_image_sitk, final_transform, 
            sitk.sitkLinear, 0.0, moving_image_sitk.GetPixelID()
        )

        # Print the cost and calculated affine transform of the registration
        print(registration_method.GetMetricValue())
        print(sitk.CompositeTransform(final_transform).GetBackTransform())

        # Save the transformation matrix
        sitk.WriteTransform(
            sitk.CompositeTransform(final_transform).GetBackTransform(), 
            outPath + moving_name[0:-5] + ".tfm"
        )

        
