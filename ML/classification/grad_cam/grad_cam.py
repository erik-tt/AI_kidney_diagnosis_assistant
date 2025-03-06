#use a trained model
from matplotlib import pyplot as plt
from monai.networks.nets import TorchVisionFCModel
from monai.visualize import GradCAMpp
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureType,
    NormalizeIntensity,
    ToTensor,
    RepeatChannel,
    Resize,
)
import numpy as np
import torch


def use_grad_CAMpp(model_path:str, inference_image_path:str, transforms:str, post_transforms:str):

    #Change model to match the model path
    model = TorchVisionFCModel(
        model_name='resnet18',
        num_classes=5,
        pretrained=True
    )

    #Load a trained model and set it to inference mode (fix file path later)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))["model_state_dict"])
    model.eval()

    tensor = transforms(inference_image_path).unsqueeze(0)

    #Run inference
    output = model(tensor)
    output = torch.argmax(output, dim=1).item()


    grad_cam = GradCAMpp(nn_module=model, target_layers="features.7.1.conv2")
    result = grad_cam(x=tensor)
    grad_cam_im = post_transforms(result[0][0].unsqueeze(0))
    img = nib.load(inference_image_path)
    img_pixel_array = np.array(img.dataobj)

    fig = plt.figure()
    im1 = plt.imshow(img_pixel_array.squeeze(0), cmap="gray")
    im2 = plt.imshow(grad_cam_im.squeeze(0) , cmap="jet", alpha=0.3)
    plt.show()




#As we use pretrained we have to resize it back to 128
transforms = Compose([
            LoadImage(),
            EnsureType(),
            RepeatChannel(repeats=3),
            Resize(spatial_size=(224, 224)),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            ToTensor()
        ])

post_transforms = Compose([
    Resize(spatial_size=(128, 128)),
])


use_grad_CAMpp(
    model_path="../classifiers/classification_models/checkpoint_resnet18.pth",
    inference_image_path="../../../data/dataset/drsprg/post/drsprg_049_POST/drsprg_049_POST_image.nii.gz",
    transforms=transforms,
    post_transforms=post_transforms
)



