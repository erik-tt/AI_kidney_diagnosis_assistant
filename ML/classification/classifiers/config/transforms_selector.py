import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    Randomizable,
    RepeatChanneld,
    Resized,
    EnsureChannelFirstd,
    Pad,
    Transposed,
    Lambdad,
    RandRotated,
    RandCoarseDropoutd,
    RandZoomd,
    ScaleIntensityd,
    RandHistogramShiftd,
    SpatialPadd
)
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, RandFlipd, RandRotate90d, 
    RandGaussianNoised, RandAdjustContrastd, RandAffined
)
from monai.transforms import RandAdjustContrast, RandHistogramShift
from monai.transforms import RandGaussianNoise, RandBiasField
from monai.transforms import Rand3DElasticd

import numpy as np
from monai.transforms import RandFlip, RandRotate, RandAffine
from monai.transforms import ScaleIntensity, NormalizeIntensity, EnsureChannelFirst
from monai.transforms import (
    ScaleIntensityd, EnsureChannelFirstd, RandFlipd, RandRotated, RandAffined, RandGaussianNoised, RandBiasFieldd
)
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PRE_TRANSFORMS = [
    LoadImaged(keys=["image"], image_only=True, reader="ITKReader"), # For DICOM?
    #EnsureTyped(keys=["image", "label"]),
    #Transposed(keys=["image"], indices=(2, 1, 0)),
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),

    #Resized(keys=["image"], spatial_size=(-1, 224, 224)),
    #Lambdad(keys=["image"], func=lambda x: x.permute(0,3,2,1)), #  Channels, Depth, Height, Width,
    #Lambdad(keys=["image"], func=lambda x: print(x.shape)),
] 

torchvision_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#Lambdad(keys=["image"], func=lambda x: print(x.shape)),
#RepeatChanneld(keys=["image"], repeats=3),
#Resized(keys=["image"], spatial_size=(-1, 224, 224))
#NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),  # Per-channel normalization
#Lambdad(keys="image", func=lambda x: torchvision_normalize(x)),  # Apply ImageNet mean/std
POST_TRANSFORMS = [
    ToTensord(keys=["image", "label"]),
]

def transforms_selector(transforms_name :str):

    transforms = []
    if transforms_name == "config_1":
        transforms = [
            ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),  # Per-channel normalization
            #SpatialPadd(keys=["image"], spatial_size=(128, 128, 180)),
            #Resized(keys="image", spatial_size=(224, 224, -1)),
            RandFlipd(keys="image", spatial_axis=2, prob=0.5),
            #RandCoarseDropoutd( #decent
            #    keys=["image"], 
            #    holes=5,  # Number of cutout regions
            #    spatial_size=(10, 10, 10),  # Size of each cutout block (3D)
            #    dropout_holes=True,  # If True, replaces with zeros
            #    fill_value=0,  # Replace cutout regions with zero intensity
            #    prob=0.5  # Apply with 50% probability
            #),
            #RandGaussianNoised(keys=["image"], mean=0.0, std=0.02, prob=0.2), #Jait
            #RandAdjustContrastd(keys=["image"], gamma=(0.9, 1.1), prob=0.5), #JAIT
            #RandAffined( ## DEECENT
            #    keys=["image"], 
            #    rotate_range=(0.1, 0.1, 0.1),  # Small rotation in 3D
            #    translate_range=(5, 5, 5),  # Small shift
            #    scale_range=(0.1, 0.1, 0.1),  # Slight zoom in/out
            #    prob=0.5
            #),
        ]

    if transforms_name == "pretrained":
        transforms = [ 
            RepeatChanneld(keys=["image"], repeats=3),
            Lambdad(keys=["image"], func=lambda x: x.permute(0,3,2,1)), # USIKKER PÅ LAMBDAFUNCTION, ER LITT RAR
            Resized(keys=["image"], spatial_size=[-1, 224, 224]), 
            #Resized(keys=["image"], spatial_size=[120, -1, -1]), 
            #RandFlipd(keys=["image"], spatial_axis=2, prob=0.5),
            #ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0), 
            #NormalizeIntensityd(keys="image"),
            
            # LITT USIKKER PÅ DENNE
            # PRØV EGET DATASET, KOMMER ANN PÅ OM MAN SKAL FINETUNE
            NormalizeIntensityd(
            keys=["image"],
            subtrahend=IMAGENET_MEAN,  # Mean subtraction (per channel)
            divisor=IMAGENET_STD,  # Standard deviation normalization (per channel)
            channel_wise=True
            ), 
        ]

    if transforms_name == "3dtransforms":
        transforms = [
            Resized(keys=["image"], spatial_size=(120, 224, 224)),
            RandFlipd(keys=["image"], spatial_axis=2, prob=0.5),
        ]
    if transforms_name == "lstm":
        transforms = [
            #The dicom images does not have a first channel
            #EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            #Lambdad function needed to switch dimensions, permute suggested by chat GPT
            #Lambdad(keys=["image"], func=lambda x: x.permute(0,2,1,3)),
            #Resize temporal dimension and image dimension
            Resized(keys=["image"], spatial_size=(120, 84, 84)),
        ]

    train_transforms = PRE_TRANSFORMS + transforms + POST_TRANSFORMS
    val_transforms = []
    
    for transform in train_transforms:
        if not isinstance(transform, Randomizable):
            val_transforms.append(transform)
    
    return Compose(train_transforms), Compose(val_transforms)
    
