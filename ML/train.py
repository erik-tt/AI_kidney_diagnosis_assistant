import config.models
from utils.file_reader import FileReader
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from sklearn.model_selection import train_test_split
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import config.models as models
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    Lambdad,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandZoomd,
    NormalizeIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    ToTensord,
    AsDiscrete
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_reader = FileReader("../data")

# Adult dataset
segmentation_data_drsprg = file_reader.get_segmentation_file_paths("drsprg/post")
#Children dataset
segmentation_data_drsbru = file_reader.get_segmentation_file_paths("drsbru/post")

segmentation_data = segmentation_data_drsprg + segmentation_data_drsbru

#Split tran test data
train_data, test_data = train_test_split(
    segmentation_data,
    test_size=0.2,      
    random_state=42,     
    shuffle=True         
)

#Fix to adjust for unkown remapping of labels
def remap_labels(label):
    label_mapping = {0: 0, 38: 1, 75: 2}
    remapped_label = label.clone()
    for orig, target in label_mapping.items():
        remapped_label[label == orig] = target
    return remapped_label


#Transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Lambdad(keys="label", func=remap_labels),           # Fix
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"])
    ]
)


val_transforms = Compose(
     [
        LoadImaged(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Lambdad(keys="label", func=remap_labels),           # Fix
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"])
    ]
)

# Create dataset
train_dataset = CacheDataset(train_data, train_transforms)
test_dataset = CacheDataset(test_data, val_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

#Define model
model = models.unet

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters())
dice_metric = DiceMetric(include_background=True, reduction="mean")
max_epochs = 120
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)


# Training loop
for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        training_losses = []
        
        for batch_data in tqdm(train_dataloader):
            images, labels = batch_data["image"].to(device), batch_data["label"].to(device) 
            optimizer.zero_grad()
            outputs = model(images)            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())

        validation_losses = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                    images, labels = batch["image"].to(device), batch["label"].to(device)

                    outputs = model(images)

                    loss = loss_function(outputs, labels)
                    validation_losses.append(loss.item())
        print(np.mean(training_losses))
        print(np.mean(validation_losses))


