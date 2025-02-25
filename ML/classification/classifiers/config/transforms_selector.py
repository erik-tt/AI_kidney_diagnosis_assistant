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
    RandZoomd
)
import numpy as np

import torchvision.transforms as transforms


PRE_TRANSFORMS = [
    LoadImaged(keys=["image"], image_only=True, reader="ITKReader"), # For DICOM?
    EnsureTyped(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    Lambdad(keys=["image"], func=lambda x: x.permute(0,3,2,1)), #  Channels, Depth, Height, Width,
] 

torchvision_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

POST_TRANSFORMS = [
    NormalizeIntensityd(keys="image"),
    ToTensord(keys=["image", "label"]),
    #Lambdad(keys="image", func=lambda x: torchvision_normalize(x)) 
    #NormalizeIntensityd(keys=["image"], subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225], nonzero=True),  # ImageNet normalization
]

def transforms_selector(transforms_name :str):

    transforms = []
    if transforms_name == "config_1":
        transforms = [
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        ]

    if transforms_name == "pretrained":
        transforms = [ 
            #Lambdad(keys=["image"], func=lambda x: print(x.shape)),
            RepeatChanneld(keys=["image"], repeats=3),
            Resized(keys=["image"], spatial_size=(-1, 224, 224))
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
    
