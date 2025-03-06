from monai.networks.nets import SwinUNETR, UNet, UNETR, AttentionUnet
import torch

from ML.segmentation.config.original_unet import Original_UNet

def model_selector(model_name :str, device: torch.device):

    #Same architecture like in Ronneberger et al (2015), but with strided convolutions as implemented in monai instead of pooling layers. Only one conv layer per block
    if model_name.lower() =="originalunet":
        return Original_UNet().to(device)

    if model_name.lower() == "baseline":
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(64,128,256, 512, 1024),
            strides=(2, 2, 2, 2),
            act=("relu", {"inplace": True}),
            dropout=0.0,
            norm=None
        ).to(device)

    #UNet with instance normalization and parametric rectified linear unit (PReLU) using instance normalization
    elif model_name.lower() == "unet":
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(64,128,256, 512, 1024),
            strides=(2, 2, 2, 2),
            dropout=0.0
        ).to(device)
    
    #Unet with residual blocks
    elif model_name.lower() == "resunet":
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(64,128,256, 512, 1024),
            strides=(2, 2, 2, 2),
            dropout=0.0,
            num_res_units=1
        ).to(device)
    
    elif model_name.lower() == "attentionunet":
        return AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(64,128,256, 512, 1024),
            strides=(2, 2, 2, 2),
            dropout=0.0,
        ).to(device)

   #Transformer based architectures
    
    #Swin transformer as encoder and CNN as decoder
    elif model_name.lower() == "swinunetr":
        return SwinUNETR(
            img_size=[128, 128],
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
        ).to(device)

    #Unet transformer, Vision transformer as encoder and CNN as decoder
    elif model_name.lower() == "unetr":
        return UNETR(
            spatial_dims=2,
            in_channels = 1,
            out_channels= 2,
            img_size = [128, 128],
        ).to(device)
    
    else:
        raise ValueError(f"Unkown model: {model_name}, spell it identical to MONAI.io")
    
