import torch
from monai.networks.nets import ResNet, EfficientNetBN
import torchvision.models as models
import torch.nn as nn

def get_mobilenetv3():
    #Testing with small model first
    mobilenet_v3 = models.mobilenet_v3_small()

    #Grayscale config (prompted chatgpt)
    mobilenet_v3.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )
    return mobilenet_v3

def model_selector(model_name :str, device: torch.device):

    if model_name.lower() == "resnet18":
        model = ResNet(
            block="basic",
            num_classes = 5,
            n_input_channels = 1,
            layers = [2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=2
        )
        return model
    
    elif model_name.lower() == "efficientnet":
        model = EfficientNetBN(
            model_name="efficientnet-b1",
            in_channels=1,
            pretrained=False
        )
        return model
    
    elif model_name.lower() == "mobilenetv3":
        return get_mobilenetv3()
    
    else:
        raise ValueError(f"Unkown classifier, check spelling of model in model_selector.py")