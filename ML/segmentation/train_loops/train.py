import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import AsDiscrete
from monai.data import DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter
import os 
from datetime import datetime

def plot_output(model_output, image, label):

    model_output = model_output.unsqueeze(0)

    model_label = torch.argmax(model_output, dim=1).squeeze(0)

    label_remove_one_hot = model_label.cpu().numpy()
    label_cpu = label.cpu().numpy()
    image_cpu = image.cpu().numpy()

    fig = plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(label_cpu[0, :, :], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Model Output")
    plt.imshow(label_remove_one_hot, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    
    return fig


def train_loop(model, 
               epochs: int, 
               train_dataloader: DataLoader, 
               val_dataloader: DataLoader, 
               device: torch.device,
               writer: SummaryWriter,
               epochs_to_save: int,
               model_name: str):

    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters())
    dice_metric = DiceMetric(include_background=False, reduction="mean")
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

                # Fra tdt17 min project
                val_labels_list = decollate_batch(labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                training_losses.append(loss.item())
                
                # Fra tdt17 mini project
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            training_dice = dice_metric.aggregate().item()
            dice_metric.reset()

            validation_losses = []
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_dataloader):
                        images, labels = batch["image"].to(device), batch["label"].to(device)

                        outputs = model(images)

                        # Fra tdt17 mini project
                        val_labels_list = decollate_batch(labels)
                        val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                        val_outputs_list = decollate_batch(outputs)
                        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

                        loss = loss_function(outputs, labels)
                        validation_losses.append(loss.item())

                        if (epoch + 1) % epochs_to_save == 0:
                            #Log the histograms of model weights
                            for name, param in model.named_parameters():
                                    writer.add_histogram(name, param, epoch)

                            writer.add_figure("ground truth vs output",
                                plot_output(outputs[0], images[0], labels[0]),
                                global_step = epoch)

                            #Save checkpoint
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss
                                },f"segmentation_models/checkpoint_{model_name}.pth")
                        
                        # TDT 17 mini project
                        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                validation_dice = dice_metric.aggregate().item()
                dice_metric.reset()

            writer.add_scalar("Training loss", np.mean(training_losses), epoch)
            writer.add_scalar("Validation loss", np.mean(validation_losses), epoch)
            writer.add_scalar("Training dice", training_dice, epoch)
            writer.add_scalar("Validation dice", validation_dice, epoch)
    writer.flush()