import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
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
from monai.data import DataLoader

def train_loop(model, 
               train_dataloader: DataLoader, 
               val_dataloader: DataLoader, 
               device: torch.device):

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


