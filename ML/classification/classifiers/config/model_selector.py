import torch
from monai.networks.nets import ResNet



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
        print(model)
        return model
    
    else:
        raise ValueError(f"Unkown classifier, check spelling of model in model_selector.py")