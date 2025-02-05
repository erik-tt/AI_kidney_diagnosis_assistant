import torch
import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    Lambdad,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    Randomizable,
    GaussianSmoothd,
    RandRotated,
    RandZoomd,
    RandCoarseDropoutd,
    RandCoarseShuffle,
    HistogramNormalized,
    RandHistogramShift,
    RandAffine,
    RandWeightedCropd,
    RandCropByPosNegLabeld,
    RandHistogramShiftd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandCoarseShuffled
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
    if transforms_name == "baseline":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        ]
    elif transforms_name == "config_1":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            GaussianSmoothd(keys=["image"], sigma=1.0),
        ]
    elif transforms_name == "config_2":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandZoomd(keys=["image"], prob=0.5, min_zoom=0.9, max_zoom=1.1),
        ]
    elif transforms_name == "config_3":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandRotated(keys=["image"], range_x= np.pi / 12, prob=0.5),
        ]
    elif transforms_name == "config_4":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandCoarseDropoutd(
                keys=["image", "label"], 
                holes=1, 
                spatial_size=(20, 20), 
                dropout_holes=True,  
                fill_value=0.0,  
                prob=0.1
            ),
        ]
    elif transforms_name == "config_5":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandCoarseDropoutd(
                keys=["image", "label"], 
                holes=5, 
                spatial_size=(20, 20), 
                dropout_holes=True,  
                fill_value=0.0,  
                prob=0.1
            ),
        ]
    elif transforms_name == "config_6":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandCoarseDropoutd(
                keys=["image", "label"], 
                holes=5, 
                spatial_size=(10, 10), 
                dropout_holes=True,  
                fill_value=0.0,  
                prob=0.1
            ),
        ]
    elif transforms_name == "config_7":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandCoarseShuffled(keys=["image"], holes=1, spatial_size=10),
        ]
    elif transforms_name == "config_8":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandCoarseShuffled(keys=["image"], holes=5, spatial_size=10),
        ]
    elif transforms_name == "config_9":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            HistogramNormalized(keys='image', num_bins=256, min=0.0, max=1.0),
        ]
    elif transforms_name == "config_10":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.5),            
        ]
    elif transforms_name == "config_11":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandCoarseDropoutd(keys=["image", "label"], holes=5, spatial_size=10),
        ]
    elif transforms_name == "config_12":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandAffine(shear_params=(0.5, 0.5), padding_mode='zeros'),
        ]
    elif transforms_name == "config_13":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandWeightedCropd(keys=["image", "label"], spatial_size=(128, 128), num_samples=4, weight_key="image"),
        ]
    elif transforms_name == "config_14":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=(128, 128),
                label_key="label",
                pos=1,  
                neg=1,   
                num_samples=4,  
            ),
        ]
    elif transforms_name == "config_15":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            HistogramNormalized(keys='image', num_bins=10, min=0.0, max=1.0),
        ]
    elif transforms_name == "config_16":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandGaussianSmoothd(keys=["image"], sigma=1.0),
        ]
    elif transforms_name == "config_17":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandGaussianSmoothd(keys=["image"]),
        ]
    elif transforms_name == "config_18":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandGaussianSharpend(keys=["image"]),
        ]
    elif transforms_name == "config_19":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandCoarseDropoutd(keys=["image", "label"], holes=5, spatial_size=10, prob=0.5),
        ]
    elif transforms_name == "config_20":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            GaussianSmoothd(keys=["image"], sigma=1.0),
            RandCoarseDropoutd(keys=["image", "label"], holes=5, spatial_size=10),
        ]
    elif transforms_name == "config_21":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            GaussianSmoothd(keys=["image"], sigma=1.0),
            RandCoarseDropoutd(keys=["image", "label"], holes=5, spatial_size=10, prob=0.5),
        ]
    elif transforms_name == "config_22":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            GaussianSmoothd(keys=["image"], sigma=1.0),
            RandCoarseDropoutd(keys=["image", "label"], holes=10, spatial_size=10),
        ]
    elif transforms_name == "config_23":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            GaussianSmoothd(keys=["image"], sigma=1.0),
            RandCoarseDropoutd(keys=["image", "label"], holes=1, spatial_size=10),
        ]
    elif transforms_name == "config_24":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            GaussianSmoothd(keys=["image"], sigma=1.0),
            RandCoarseDropoutd(keys=["image", "label"], holes=5, spatial_size=20),
        ]
    elif transforms_name == "config_25":
        transforms = [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            GaussianSmoothd(keys=["image"], sigma=1.0),
            RandCoarseDropoutd(keys=["image", "label"], holes=5, spatial_size=5),
        ]


    train_transforms = PRE_TRANSFORMS + transforms + POST_TRANSFORMS
    val_transforms = []

    for transform in train_transforms:
        if not isinstance(transform, Randomizable):
            val_transforms.append(transform)
    
    return Compose(train_transforms), Compose(val_transforms)
    
