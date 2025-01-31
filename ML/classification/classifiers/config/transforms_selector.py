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
    Lambdad
)

PRE_TRANSFORMS = [
    LoadImaged(keys=["image"]),
    EnsureTyped(keys=["image", "label"]),
] 

POST_TRANSFORMS = [
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"])
]

def transforms_selector(transforms_name :str):

    transforms = []
    if transforms_name == "config_1":
        transforms = [
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        ]

    if transforms_name == "pretrained":
        transforms = [ 
            #Lambdad(keys=["image"], func=lambda x: print(x.shape)),
            RepeatChanneld(keys=["image"], repeats=3),
            Resized(keys=["image"], spatial_size=(224, 224))
        ]

    if transforms_name == "3dtransforms":
        transforms = [
            #The dicom images does not have a first channel
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            #Lambdad function needed to switch dimensions, permute suggested by chat GPT
            Lambdad(keys=["image"], func=lambda x: x.permute(0,2,1,3)),
            #Resize temporal dimension and image dimension
            Resized(keys=["image"], spatial_size=(48, 48, 48)),
        ]

    train_transforms = PRE_TRANSFORMS + transforms + POST_TRANSFORMS
    val_transforms = []

    for transform in train_transforms:
        if not isinstance(transform, Randomizable):
            val_transforms.append(transform)
    
    return Compose(train_transforms), Compose(val_transforms)
    
