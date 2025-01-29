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
    Pad,
    Transposed,
    Lambdad
)

PRE_TRANSFORMS = [
    LoadImaged(keys=["image"], ensure_channel_first=True),
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

    if transforms_name == "config_2":
        transforms = [ 
            RepeatChanneld(keys=["image"], repeats=3),
            Resized(keys=["image"], spatial_size=(224, 224))
        ]

    if transforms_name == "config_3":
        transforms = [
            #Lambdad function needed to switch dimensions, suggested by chat GPT
            Lambdad(keys=["image"], func=lambda x: x.permute(0, 2, 1, 3)),

            #Resize the image to be uniform
            # 12 images with 48x48 resolution
            Resized(keys=["image"], spatial_size=(12, 48, 48))
        ]

    train_transforms = PRE_TRANSFORMS + transforms + POST_TRANSFORMS
    val_transforms = []

    for transform in train_transforms:
        if not isinstance(transform, Randomizable):
            val_transforms.append(transform)
    
    return Compose(train_transforms), Compose(val_transforms)
    
