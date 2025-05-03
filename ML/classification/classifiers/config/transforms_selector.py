from monai.transforms import (
    Compose,
    LoadImaged,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    Randomizable,
    RepeatChanneld,
    Resized,
    EnsureChannelFirstd,
    Lambdad,
    ScaleIntensityd,
    SpatialPadd,
    RandAffined
)


PRE_TRANSFORMS = [
    LoadImaged(keys=["image"], image_only=True, reader="ITKReader"),
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
] 

POST_TRANSFORMS = [
    ToTensord(keys=["image", "label"]),
]

def transforms_selector(transforms_name :str):

    transforms = []
    if transforms_name == "config_1":
        transforms = [
            ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),  # Per-channel normalization

            # FOR CUDA
            Lambdad(keys=["image"], func=lambda x: x.permute(0,3,2,1)), # SJEKKE AT DET ER RIKTIG,
            
            # FOR NOT CUDA
            #Lambdad(keys=["image"], func=lambda x: x.permute(0,2,1,3)),
            
            SpatialPadd(keys=["image"], spatial_size=(180, -1, -1)), 
            RandFlipd(keys="image", spatial_axis=1, prob=0.5),
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
            
            # FOR CUDA
            Lambdad(keys=["image"], func=lambda x: x.permute(0,3,2,1)), # SJEKKE AT DET ER RIKTIG,
            
            # FOR NOT CUDA
            #Lambdad(keys=["image"], func=lambda x: x.permute(0,2,1,3)),

            Resized(keys=["image"], spatial_size=[-1, 224, 224]), 
            ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys="image"),
            RandFlipd(keys="image", spatial_axis=1, prob=0.5),
        ]

    train_transforms = PRE_TRANSFORMS + transforms + POST_TRANSFORMS
    val_transforms = []
    
    for transform in train_transforms:
        if not isinstance(transform, Randomizable):
            val_transforms.append(transform)
    
    return Compose(train_transforms), Compose(val_transforms)
    
