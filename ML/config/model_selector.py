from monai.networks.nets import SwinUNETR, UNet
import torch

def model_selector(model_name :str, device: torch.device):

    if model_name == "UNet":
        return UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            channels=(16,32,64,128,256),
            strides=(2, 2, 2, 2),
        ).to(device)
    
    elif model_name == "SwinUNETR":
        return SwinUNETR(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            channels=(16,32,64,128,256),
            strides=(2, 2, 2, 2),
        ).to(device)
    
    else:
        raise ValueError(f"Unkown model: {model_name}, spell it identical to MONAI.io")
    
