from monai.networks.nets import UNet, SwinUNETR
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=3,
    channels=(16,32,64,128,256),
    strides=(2, 2, 2, 2),
).to(device)

swin_unetr = SwinUNETR(
    spatial_dims=2,
    in_channels=1,
    out_channels=3,
    channels=(16,32,64,128,256),
    strides=(2, 2, 2, 2),
).to(device)