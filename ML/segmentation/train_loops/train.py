import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import AsDiscrete
from monai.data import DataLoader, decollate_batch, CacheDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
import os 
from datetime import datetime
from sklearn.model_selection import KFold
from config.model_selector import model_selector
from config.transforms_selector import transforms_selector
from monai.transforms import RemoveSmallObjects, FillHoles

# Global Variables

remove_objects = RemoveSmallObjects(min_size=20, connectivity=2)
fill_holes = FillHoles(connectivity=2)

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

def train(model,
        loss_function,
        train_dataloader: DataLoader, 
        device: torch.device,
        optimizer
        ):
    
    model.train()
    training_losses = []

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    for batch_data in tqdm(train_dataloader):
        images, labels = batch_data["image"].to(device), batch_data["label"].to(device) 
        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

        # Fra tdt17 min project
        train_labels_list = decollate_batch(labels)
        train_labels_convert = [AsDiscrete()(label) for label in train_labels_list]
        train_outputs_list = decollate_batch(outputs)
        train_outputs_convert = [AsDiscrete(argmax=True)(pred) for pred in train_outputs_list]
        
        train_outputs_convert = [fill_holes(pred.to(torch.uint8)) for pred in train_outputs_convert]
        train_outputs_convert = [remove_objects(pred.to(torch.uint8)) for pred in train_outputs_convert]

        # Fra tdt17 mini project
        dice_metric(y_pred=train_outputs_convert, y=train_labels_convert)
    training_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return np.mean(training_losses), training_dice

def validate(model,
            loss_function,
            val_dataloader: DataLoader, 
            device: torch.device,
            optimizer,
            writer, 
            epoch=None, 
            epochs_to_save=None, 
            model_name=None,
            log=True
            ):
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    precision_metric = ConfusionMatrixMetric(metric_name="precision", include_background=False, reduction="mean")
    recall_metric = ConfusionMatrixMetric(metric_name="recall", include_background=False, reduction="mean")
     
    validation_losses = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
                images, labels = batch["image"].to(device), batch["label"].to(device)

                outputs = model(images)

                loss = loss_function(outputs, labels)
                validation_losses.append(loss.item())

                # Fra tdt17 mini project
                val_labels_list = decollate_batch(labels)
                val_labels_convert = [AsDiscrete()(label) for label in val_labels_list]
                val_outputs_list = decollate_batch(outputs)
                val_outputs_convert = [AsDiscrete(argmax=True)(pred) for pred in val_outputs_list]

                val_outputs_convert = [fill_holes(pred.to(torch.uint8)) for pred in val_outputs_convert]
                val_outputs_convert = [remove_objects(pred.to(torch.uint8)) for pred in val_outputs_convert]

                if log and (epoch + 1) % epochs_to_save == 0:
                    #Log the histograms of model weights
                    for name, param in model.named_parameters():
                            writer.add_histogram(name, param, epoch)

                    writer.add_figure("ground truth vs output",
                        plot_output(outputs[0], images[0], labels[0]), #val outputs convert aswell
                        global_step = epoch)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        },f"segmentation_models/checkpoint_{model_name}.pth")
                    
                
                # TDT 17 mini project
                dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                iou_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                precision_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                recall_metric(y_pred=val_outputs_convert, y=val_labels_convert)

        #Take mean for every batch
        validation_dice = dice_metric.aggregate().item()
        validation_iou = iou_metric.aggregate().item()
        validation_precision = precision_metric.aggregate()[0].item()
        validation_recall = recall_metric.aggregate()[0].item()

        #Reset the metrics
        dice_metric.reset()
        iou_metric.reset()

        return np.mean(validation_losses), validation_dice, validation_iou, validation_precision, validation_recall


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
    best_dice = -1
    best_dice_epoch = -1

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        
        training_loss, training_dice = train(model, 
                                                loss_function, 
                                                train_dataloader,
                                                device,
                                                optimizer,
                                                )
        
        validation_loss, validation_dice, validation_iou, validation_precision, validation_recall = validate(model, 
                                                loss_function, 
                                                val_dataloader,
                                                device,
                                                optimizer,
                                                writer,
                                                epoch,
                                                epochs_to_save,
                                                model_name,
                                                )
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_dice_epoch = epoch + 1
            #Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },f"segmentation_models/best_{model_name}.pth")
            
            writer.add_scalar("best dice", best_dice)
            writer.add_scalar("best dice epoch", best_dice_epoch)

        
        print(f"Training loss: {np.mean(training_loss)}")
        print(f"Validation loss: {np.mean(validation_loss)}")
        print(f"Training dice: {training_dice}")
        print(f"Validation dice: {validation_dice}")
        print(f"Validation IoU: {validation_iou}")
        print(f"Validation precision: {validation_precision}")
        print(f"Validation recall: {validation_recall}")
        print(f"Best dice: {best_dice}")
        print(f"Best dice epoch: {best_dice_epoch}")
        
        writer.add_scalar("Training loss", np.mean(training_loss), epoch)
        writer.add_scalar("Validation loss", np.mean(validation_loss), epoch)
        writer.add_scalar("Training dice", training_dice, epoch)
        writer.add_scalar("Validation dice", validation_dice, epoch)
    writer.flush()


def k_fold_validation(model_name,
                      dataset, 
                      epochs:int, 
                      batch_size: int, 
                      device: torch.device,
                      writer: SummaryWriter,
                      transforms_name: str,
                      num_workers: int,
                      splits: int = 5):
    
    
    kfold = KFold(n_splits=splits, shuffle=True, random_state=42)

    train_transforms, val_transforms = transforms_selector(transforms_name)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        
        #Need to reinitalize the model every time
        model = model_selector(model_name, device)
        loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters())

        print(f"Fold {fold+1}/{splits}")

        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]

        train_ds = CacheDataset(train_set, train_transforms)
        val_ds = CacheDataset(val_set, val_transforms)

        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for epoch in range(epochs):
             print("-" * 10)
             print(f"epoch {epoch + 1}/{epochs}")
             training_loss, training_dice = train(model, 
                                                loss_function, 
                                                train_dataloader,
                                                device,
                                                optimizer)
        
        validation_loss, validation_dice, validation_iou, validation_precision, validation_recall = validate(model, 
                                                loss_function, 
                                                val_dataloader,
                                                device,
                                                optimizer,
                                                writer,
                                                log=False)
        
        print(f"Training loss: {np.mean(training_loss)}")
        print(f"Validation loss: {np.mean(validation_loss)}")
        print(f"Training dice: {training_dice}")
        print(f"Validation dice: {validation_dice}")
        print(f"Validation IoU: {validation_iou}")
        print(f"Validation precision: {validation_precision}")
        print(f"Validation recall: {validation_recall}")
        
        #Write the model metrics to tensorboard
        writer.add_scalar("Training loss", np.mean(training_loss), fold)
        writer.add_scalar("Validation loss", np.mean(validation_loss), fold)
        writer.add_scalar("Training dice", training_dice, fold)
        writer.add_scalar("Validation dice", validation_dice, fold)
        writer.add_scalar("Validation IoU", validation_iou, fold)
        writer.add_scalar("Validation precision", validation_precision, fold)
        writer.add_scalar("Validation recall", validation_recall, fold)
    writer.flush()

