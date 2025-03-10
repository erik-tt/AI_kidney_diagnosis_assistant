from monai.networks.nets import SwinUNETR, BasicUNet, BasicUNetPlusPlus, UNet, UNETR, AttentionUnet
import torch

# from ML.segmentation.config.original_unet import Original_UNet

def model_selector(model_name :str, device: torch.device):

    #Same architecture like in Ronneberger et al (2015), but with strided convolutions as implemented in monai instead of pooling layers. Only one conv layer per block
    # if model_name.lower() =="originalunet":
    #     return Original_UNet().to(device)

    #CNN based encoders:

    #Unet with leakyrelu and inst
    if model_name.lower() == "unet":
        return BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=(64, 128, 256, 512, 1024, 64),
            dropout=0.0,
            act=('prelu')
        ).to(device)
    
    #unet++ with leaky rely and inst norm
    if model_name.lower() == "unetpp":
        return BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            features=(64, 128, 256, 512, 1024, 64),
            act=('prelu'),
            dropout=0.0,
        ).to(device)
    
    #Unet with 2 conv layer per residual block
    elif model_name.lower() == "resunet":
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(64,128,256, 512, 1024),
            strides=(2, 2, 2, 2),
            dropout=0.0,
            num_res_units=2,
            act=('prelu')
        ).to(device)
    
    
    elif model_name.lower() == "attentionunet":
        return AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(64,128,256, 512, 1024),
            strides=(2, 2, 2, 2),
        ).to(device)

   #Transformer based architectures
    
    #Swin transformer as encoder and CNN as decoder, v2 for residual connections and more stable training
    elif model_name.lower() == "swinunetrv2":
        return SwinUNETR(
            img_size=[128, 128],
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            use_v2= True
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
    
    
