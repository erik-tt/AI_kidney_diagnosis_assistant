import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    Lambdad,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    Randomizable,
    RandGaussianSmoothd
)

def remap_labels(label):
    label_mapping = {0: 0, 38: 1, 75: 1}
    remapped_label = label.clone()
    for orig, target in label_mapping.items():
        remapped_label[label == orig] = target
    return remapped_label

PRE_TRANSFORMS = [
    LoadImaged(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    Lambdad(keys="label", func=remap_labels), #FIX
] 

POST_TRANSFORMS = [
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"])
]

def transforms_selector(transforms_name :str):

    transforms = []
    if transforms_name == "config_1":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandGaussianSmoothd(keys=["image"])
        ]


    train_transforms = PRE_TRANSFORMS + transforms + POST_TRANSFORMS
    val_transforms = []

    for transform in train_transforms:
        if not isinstance(transform, Randomizable):
            val_transforms.append(transform)
    
    return Compose(train_transforms), Compose(val_transforms)
    
