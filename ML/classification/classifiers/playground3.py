from typing import List
from sklearn.model_selection import train_test_split
from config.transforms_selector import transforms_selector
from config.model_selector import model_selector
from monai.data import CacheDataset, Dataset, DataLoader
from ML.utils.file_reader import get_classification_data
from ML.classification.classifiers.dataset.ClassificationDataset import ClassificationDataset
from monai.data import DataLoader, pad_list_data_collate

# REmove
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    Randomizable,
    RepeatChanneld,
    Resized,
    EnsureChannelFirstd,
    Pad,
    Transposed,
    Lambdad,
    RandRotated,
    RandCoarseDropoutd,
    RandZoomd,
    ScaleIntensityd
)
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, RandFlipd, RandRotate90d, 
    RandGaussianNoised, RandAdjustContrastd, RandAffined
)

import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch
import pandas as pd
import pydicom

feature_extractor = models.resnet18(pretrained=True)
#feature_extractor2 = ResNetFeatures("resnet50", pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Remove the final fully connected layer
feature_extractor.fc = torch.nn.Identity()  # Replace fc layer with an identity layer

# Set the model to evaluation mode
feature_extractor.eval()
### **🚀 Step 1: Load Dataset**
dataset = get_classification_data(["drsprg"], ["POST"], radiomics=False)

all_features, all_labels, all_images = [], [], []
names = None  # Store feature names

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


PRE_TRANSFORMS = [
    LoadImaged(keys=["image"], image_only=True, reader="ITKReader"), # For DICOM?
    #EnsureTyped(keys=["image", "label"]),
    #Transposed(keys=["image"], indices=(2, 1, 0)),
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    RepeatChanneld(keys=["image"], repeats=3),
    #Resized(keys=["image"], spatial_size=[120, -1, -1]), 
    #RandFlipd(keys=["image"], spatial_axis=2, prob=0.5),
    #ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0), 
    #NormalizeIntensityd(keys="image"),
    
    # LITT USIKKER PÅ DENNE
    # PRØV EGET DATASET, KOMMER ANN PÅ OM MAN SKAL FINETUNE 
    NormalizeIntensityd(
       keys=["image"],
       subtrahend=IMAGENET_MEAN,  # Mean subtraction (per channel)
       divisor=IMAGENET_STD,  # Standard deviation normalization (per channel)
       channel_wise=True
    ), 
    ToTensord(keys=["image", "label"]),

    #Resized(keys=["image"], spatial_size=(-1, 224, 224)),
    #Lambdad(keys=["image"], func=lambda x: x.permute(0,3,2,1)), #  Channels, Depth, Height, Width,
    #Lambdad(keys=["image"], func=lambda x: print(x.shape)),
] 

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to 3-channel
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet stats
    transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension (shape: [1, 3, H, W])
])
transforms = Compose(PRE_TRANSFORMS)
model = model_selector("cnn", device)
data = ClassificationDataset(dataset, 0, None, "time_series", False, False, True, transforms)
bs = 4
dl = DataLoader(data, batch_size=bs, shuffle=True, num_workers=0)

model.eval()
all_features = []
all_labels = []

for batch in dl:
    img, label = batch["image"].to(device), batch["label"].to(device)
    output, features = model(img)
    all_features.append(features.cpu().numpy().reshape(bs, 180, 512))  # Shape: (batch_size, 180, 512)
    all_labels.append(label.cpu().numpy())  # Store labels too if needed

all_features = np.concatenate(all_features, axis=0)  # Final shape: (total_samples, 180, 512)
all_labels = np.concatenate(all_labels, axis=0)  # Shape: (total_samples,)

np.save("features_drsprg_model.npy", all_features)  # Saves feature matrix
np.save("labels_drsprg_model.npy", all_labels)  # Saves labels
print(all_features.shape, all_labels.shape)

j = -1
all_f = []
all_labels = []
# **Extract Features & Labels**
for entry in dataset:
    j += 1

    print(j)
    if j == 1:
        break

    label = entry["label"]
    image_path = entry["image"]
    
    if pd.isna(label):
        print("skip")
        continue  # Move to the next entry
    
    dicom_data = pydicom.dcmread(image_path)
    pixel_array = dicom_data.pixel_array  # Shape: (frames, height, width) or (height, width)

    counter = pixel_array.shape[0]
    image_features_list = []
    
    #print(j)
    if counter != 180:
        print(image_path)
        continue
    for i in range(counter):
        try:
            pix = pixel_array[i, :, :].astype(np.float32)
            
            #mean_image = np.mean(pixel_array, axis=0)  # Shape: (height, width)
            mean_image_pil = Image.fromarray(pix)

            image_tensor = preprocess(mean_image_pil)
            image_tensor = image_tensor.to(torch.float32)  # Ensure tensor is float
            with torch.no_grad():  # Disable gradient computation
                image_features = feature_extractor(image_tensor).squeeze().numpy()  # Shape: (512,)
            if image_features is not None:
                image_features_list.append(image_features)
            else:
                print(image_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            image_features = None  # Default to None or handle it appropriately
        #print(np.array(image_features_list).shape)
    time_series_features = np.array(image_features_list)
    print(time_series_features)
    all_f.append(time_series_features)
    all_labels.append(label)

all_f = np.array(all_f)
all_labels = np.array(all_labels, dtype=int)  # Ensure numeric
#np.save("features_drsprg2.npy", all_f)  # Saves feature matrix
#np.save("labels_drsprg2.npy", all_labels)  # Saves labels