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
    RandAffined,
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
            # FOR NOT CUDA

            Lambdad(keys=["image"], func=lambda x: x.permute(0,2,1,3)),
            
            SpatialPadd(keys=["image"], spatial_size=(180, -1, -1), method="end"),
            
            NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),

            # FOR CUDA
            #Lambdad(keys=["image"], func=lambda x: x.permute(0,3,2,1)), # SJEKKE AT DET ER RIKTIG,
            
            RandFlipd(keys="image", spatial_axis=2, prob=0.5),
        ]

    if transforms_name == "pretrained":
        transforms = [ 
            Lambdad(keys=["image"], func=lambda x: x.permute(0,2,1,3)),
            
            NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),

            RepeatChanneld(keys=["image"], repeats=3),
            # FOR CUDA
            #Lambdad(keys=["image"], func=lambda x: x.permute(0,3,2,1)), # SJEKKE AT DET ER RIKTIG,
            
            # FOR NOT CUDA

            Resized(keys=["image"], spatial_size=[-1, 224, 224]), 
            RandFlipd(keys="image", spatial_axis=2, prob=0.5),
        ]

    train_transforms = PRE_TRANSFORMS + transforms + POST_TRANSFORMS
    val_transforms = []
    
    for transform in train_transforms:
        if not isinstance(transform, Randomizable):
            val_transforms.append(transform)
    
    return Compose(train_transforms), Compose(val_transforms)
    
