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
    Resized
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

    if transforms_name == "config_2":
        transforms = [ 
            RepeatChanneld(keys=["image"], repeats=3),
            Resized(keys=["image"], spatial_size=(224, 224))
        ]
    train_transforms = PRE_TRANSFORMS + transforms + POST_TRANSFORMS
    val_transforms = []

    for transform in train_transforms:
        if not isinstance(transform, Randomizable):
            val_transforms.append(transform)
    
    return Compose(train_transforms), Compose(val_transforms)
    
