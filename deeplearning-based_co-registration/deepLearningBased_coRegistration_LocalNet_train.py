# -*- coding: utf-8 -*-
"""
Deep learning-based deformable co-registration of mesoscopic photoacoustic images.
Fixed and moving paired intensity and segmented images are given as an input for predicting the deformation field aligning the moving spatial domain to the fixed one. 
Weak supervision is done by minimising normalised cross-correlation (NCC) loss calculated on paired intensity images, generalised Dice loss on segmentations, and bending energy loss on the output deformation field.

Not for clinical use.
SPDX-FileCopyrightText: 2024 Cancer Research UK Cambridge Institute, University of Cambridge, Cambridge, UK
SPDX-FileCopyrightText: 2024 Thierry L. Lefebvre
SPDX-FileCopyrightText: 2024 Sarah E. Bohndiek
SPDX-License-Identifier: MIT
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, Dataset
from monai.losses import BendingEnergyLoss, LocalNormalizedCrossCorrelationLoss, GeneralizedDiceLoss
from monai.metrics import GeneralizedDiceScore
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandAffined,
    ToTensord,
)
from monai.utils import set_determinism

def forward(batch_data, model):
    """
    Forward pass through the model.

    Parameters:
    batch_data (dict): Dictionary containing fixed and moving images and labels.
    model (torch.nn.Module): The network model.

    Returns:
    tuple: Deformation field, predicted image, and predicted label.
    """
    fixed_image = batch_data["fixed_image"].to(device)
    moving_image = batch_data["moving_image"].to(device)
    moving_label = batch_data["moving_label"].to(device)

    # Predict deformation field (DDF)
    ddf = model(torch.cat((moving_image, fixed_image), dim=1))

    # Warp the moving image and label using the predicted DDF
    pred_image = warp_layer(moving_image, ddf)
    pred_label = warp_layer(moving_label, ddf)

    return ddf, pred_image, pred_label

# Define base path
root_dir = 'PATH TO BASE DIRECTORY'
print(root_dir)

# Paths to input images and segmentations
imgPath = 'PATH TO INTENSITY IMAGES'
segPath = 'PATH TO SEGMENTED IMAGES'

# Get the list of images and segmentations
namesnames = os.listdir(imgPath)
segList = os.listdir(segPath)

# List of grouped images with specific IDs to process
IDList = np.array(['INSERT LIST OF IDENTIFIERS GROUPING IMAGES TO CO-REGISTER'])

# Initialise arrays to store fixed/moving images/segmentations filenames
fixed_names = []
moving_names = []
fixed_names_seg = []
moving_names_seg = []

# Generate pairs of fixed and moving images and corresponding segmentations
for ID in IDList:
    namesID = [matchin for matchin in namesnames if ID in matchin]
    
    # Load fixed image (reference image for co-registration)
    fixed_name = namesID[0]
    namesID = namesID[1:]
    
    namesIDseg = [match for match in segList if ID in match]
    fixed_name_seg = namesIDseg[0]
    namesIDseg = namesIDseg[1:]    

    for iterseg, moving_name in enumerate(namesID):
        fixed_names.append(imgPath + fixed_name)
        fixed_names_seg.append(segPath + fixed_name_seg)
        moving_names.append(imgPath + moving_name)
        moving_name_seg = namesIDseg[iterseg]
        moving_names_seg.append(segPath + moving_name_seg)

# Create a DataFrame to store the file paths
data_df = pd.DataFrame(
    {
        "fixed_image": fixed_names,
        "moving_image": moving_names,
        "fixed_label": fixed_names_seg,
        "moving_label": moving_names_seg,
    },
)

# Set a deterministic seed for reproducibility
random_seed = 42
set_determinism(seed=random_seed)
np.random.seed(random_seed)

# Split the data into training and validation sets
split_ratio=0.5
indexes = np.arange(len(fixed_names))
train_indexes, val_indexes = train_test_split(
    indexes, train_size=split_ratio, random_state=random_seed
)
datadf_train = data_df.iloc[[train_indexes]]
datadf_val = data_df.iloc[[val_indexes]]

# Resulting training/validation file lists for Dataset and DataLoader
train_files, val_files = datadf_train.to_dict('records'), datadf_val.to_dict('records')

# Define transformations for the training and validation datasets
train_transforms = Compose(
    [
        LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]),
        EnsureChannelFirstd(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]),
        ScaleIntensityd(keys=["fixed_image", "moving_image"], minv=0., maxv=100.),
        RandAffined(
            keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
            mode=("bilinear", "bilinear", "nearest", "nearest"),
            prob=1.0,
            spatial_size=(256, 256, 64),
            rotate_range=(0, 0, np.pi / 15),
        ),        
        ToTensord(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]),
        EnsureChannelFirstd(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]),
        ScaleIntensityd(keys=["fixed_image", "moving_image"], minv=0., maxv=100.),   
        ToTensord(keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]),
    ]
)

# Create training/validation Datasets and DataLoaders
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

# Set up the model, loss functions, and optimiser
device = torch.device("cuda:0")
model = LocalNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=3,
    num_channel_initial=32,
    extract_levels=[3],
    out_activation=None,
    out_kernel_initializer="kaiming_uniform",
).to(device)

warp_layer = Warp(mode='nearest').to(device)
image_loss = LocalNormalizedCrossCorrelationLoss()  
label_loss = GeneralizedDiceLoss()  
regularization = BendingEnergyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# Quality metric to assess during validation
dice_metric = GeneralizedDiceScore()

# Initialize training parameters
max_epochs = 1000
val_interval = 1
best_metric = -1
best_metric_epoch = -1
best_metric_haus = 10000
best_epoch_loss = 10000
best_metric_haus_epoch = -1
epoch_loss_values = []
metric_values = []
img_loss_values = []
label_loss_values = []

# Load pre-trained weights
dst = 'PATH TO PRE-TRAINED WEIGHTS (IF APPLICABLE)'
model.load_state_dict(torch.load(dst))

# Training loop
for epoch in range(max_epochs):
    if (epoch + 1) % val_interval == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_ddf, val_pred_image, val_pred_label = forward(val_data, model)
                val_pred_label[val_pred_label > 1] = 1

                val_fixed_image = val_data["fixed_image"].to(device)
                val_fixed_label = val_data["fixed_label"].to(device)
                dice_metric(y_pred=val_pred_label, y=val_fixed_label)

            # Generalized Dice quality metric
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_dice_model.pth"))
                print("saved new best Dice metric model")
            print(
                f"current epoch: {epoch + 1} "
                f"current mean dice: {metric:.4f}\n"                                
                f"best mean dice: {best_metric:.4f}\n"
                f"at epoch: {best_metric_epoch}\n")
    
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()

    epoch_loss = 0
    epoch_img_loss = 0
    epoch_label_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()

        ddf, pred_image, pred_label = forward(batch_data, model)
        pred_label[pred_label > 1] = 1

        fixed_image = batch_data["fixed_image"].to(device)
        fixed_label = batch_data["fixed_label"].to(device)

        # Compute losses
        imgloss = image_loss(pred_image, fixed_image)
        labelloss = label_loss(pred_label, fixed_label)
        
        loss = imgloss + labelloss + regularization(ddf)
        loss.backward()
        optimizer.step()
        
        epoch_img_loss += imgloss.item()
        epoch_label_loss += labelloss.item()
        epoch_loss += loss.item()
        
    # Compute the dice metric for the current epoch
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    metric_values.append(metric)
    
    epoch_loss /= step
    epoch_img_loss /= step
    epoch_label_loss /= step
    epoch_loss_values.append(epoch_loss)
    img_loss_values.append(epoch_img_loss)
    label_loss_values.append(epoch_label_loss)
    
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    print(f"epoch {epoch + 1} average img loss: {epoch_img_loss:.4f}")
    print(f"epoch {epoch + 1} average label loss: {epoch_label_loss:.4f}")
    
    # Save the best model based on training loss
    if epoch_loss < best_epoch_loss:
        best_epoch_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(root_dir, "best_loss_train_model.pth"))
        print("saved new best loss model")
  

# Plot the training loss and dice metric
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
plt.plot(x, epoch_loss_values, "-b", label="Total Loss")
plt.plot(x, img_loss_values, "-r", label="Cross-Correlation Loss")
plt.plot(x, label_loss_values, "-g", label="Generalized Dice Loss")
plt.legend(loc="upper right")

plt.subplot(1, 2, 2)
plt.xlabel("Epoch")
plt.ylabel("Average Generalized Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
plt.plot(x, metric_values)
plt.show()
