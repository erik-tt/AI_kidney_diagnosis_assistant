import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import AsDiscrete
from monai.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
from datetime import datetime

def plot_val_image(model_output, image, label):

    model_output = model_output.unsqueeze(0)

    model_label = torch.argmax(model_output, dim=1).squeeze(0)

    label_remove_one_hot = model_label.cpu().numpy()
    label_cpu = label.cpu().numpy()
    image_cpu = image.cpu().numpy()

    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(label_cpu[0, :, :], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Model Output")
    plt.imshow(label_remove_one_hot, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def train_loop(model, 
               epochs: int, 
               train_dataloader: DataLoader, 
               val_dataloader: DataLoader, 
               device: torch.device,
               writer: SummaryWriter):

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters())
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_label = AsDiscrete(to_onehot=3)
    post_pred = AsDiscrete(argmax=True, to_onehot=3)

    # Training loop
    for epoch in range(epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")
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

                        if (epoch + 1) % 10 == 0:
                             #writer.add_figure("ground truth vs")
                            plot_val_image(outputs[0], images[0], labels[0])
                            
    
                        
            writer.add_scalar("Training loss", np.mean(training_losses), epoch)
            writer.add_scalar("Validation loss", np.mean(validation_losses), epoch)
