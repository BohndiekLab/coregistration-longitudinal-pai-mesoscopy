import numpy as np
from skimage.measure import block_reduce
from skimage import io
from tifffile import imsave
import os
import scipy.stats as sp

def std_norm(data):
    
    # Apply z-score normalisation
    if np.std(data) > 0.:
        return (data - np.mean(data)) / np.std(data)
    else:
        return (data - np.mean(data))

def preprocess_img(img):
    
    # Max downsampling to isotropic resolution (4,20,20)um to (20,20,20)um
    img = img[0:700,:,:]
    img = block_reduce(img, block_size=(5,1,1), func=np.max)
    
    # Apply local standardisation
    for j in range(img.shape[0]):
        img[j,:,:] = std_norm(img[j,:,:])
        
    # Re-arrange intensity histogram
    lp = sp.scoreatpercentile(img,0.05)
    up = sp.scoreatpercentile(img,99.95)
    img[img < lp] = lp
    img[img > up] = up
    
    return img

def preprocess_roi(roi):
    
    # Max downsampling
    roi = roi[0:700,:,:]
    roi = block_reduce(roi, block_size=(5,1,1), func=np.max)
    
    return roi

# Paths to input images and output results
dirname = 'PATH TO IMAGES'
roidir  = 'PATH TO REGIONS OF INTEREST'
outdir  = 'PATH FOR SAVING PREPROCESSED IMAGES'

# List of filenames to process
dirfiles = os.listdir(dirname)

for name in dirfiles:

    print('Preprocessing '+name)
    namesplit = name.split('.')
    namesplit = namesplit[0]

    img_init = io.imread(dirname+"/"+name)
    roi_init = io.imread(roidir+"/"+namesplit+".tiff")

    # Preprocess intensity images and regions of interest
    img = preprocess_img(img_init)
    roi = preprocess_roi(roi_init)

    # Apply regions of interest on images
    img = img*roi

    # Export preprocessed images as TIFF
    imsave(outdir+"/"+namesplit+".tiff",img)
    print('Saved image to '+outdir+"/"+name)
    
    