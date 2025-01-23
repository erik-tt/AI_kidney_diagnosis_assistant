import numpy as np
import torch
from tqdm import tqdm
from monai.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_loop(model, 
               epochs: int, 
               train_dataloader: DataLoader, 
               val_dataloader: DataLoader, 
               device: torch.device,
               epochs_to_save: int,
               model_name: str):

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    training_losses = []
    validation_losses = []
    training_accuracy = []
    validation_accuracy = []


    # Training loop
    for epoch in range(epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")
            model.train()

            training_losses_epoch = []
    
            for batch_data in tqdm(train_dataloader):
                images, labels = batch_data["image"].to(device), batch_data["label"].to(device, dtype=torch.long) 
                #Labels should be 1 index
                labels = labels - 1
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                training_losses_epoch.append(loss.item())

            training_losses.append(np.mean(training_losses_epoch))

            
            
            correct = 0
            total = 0
            validation_labels = []
            validation_predictions = []
            validation_losses_epoch = []

            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_dataloader):
                        images, labels = batch["image"].to(device), batch["label"].to(device, dtype=torch.long) 
                        labels = labels - 1
                        outputs = model(images)

                        loss = loss_function(outputs, labels)
                        validation_losses_epoch.append(loss.item())
                        _, predicted = torch.max(outputs.data, 1)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        validation_labels.extend(labels.cpu().numpy())
                        validation_predictions.extend(predicted.cpu().numpy())

            validation_losses.append(np.mean(validation_losses_epoch))
            validation_accuracy.append(100 * correct / total)

            print(f"""Epoch {epoch+1}, Training Loss: {training_losses[-1]},
                   Validation Loss: {validation_losses[-1]}
                   Accuracy: {validation_accuracy[-1]}%""")