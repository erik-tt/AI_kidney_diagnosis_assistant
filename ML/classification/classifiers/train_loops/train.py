from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from monai.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(true_labels, predicted_labels):
    CKD_stages = np.arange(1,6)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CKD_stages, yticklabels=CKD_stages)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for validation set, last Epoch")
    plt.show()
        


def train_loop(model, 
               epochs: int, 
               train_dataloader: DataLoader, 
               val_dataloader: DataLoader, 
               device: torch.device,
               writer: SummaryWriter,
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

            training_losses = []
    
            for batch_data in tqdm(train_dataloader):
                images, labels = batch_data["image"].to(device), batch_data["label"].to(device, dtype=torch.long) 
                #Labels should be 1 index
                labels = labels - 1
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                training_losses.append(loss.item())

            
            
            correct = 0
            total = 0
            validation_labels = []
            validation_predictions = []
            validation_losses = []

            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_dataloader):
                        images, labels = batch["image"].to(device), batch["label"].to(device, dtype=torch.long) 
                        labels = labels - 1
                        outputs = model(images)

                        loss = loss_function(outputs, labels)
                        validation_losses.append(loss.item())

                        _, predicted = torch.max(outputs.data, 1)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        validation_labels.extend(labels.cpu().numpy())
                        validation_predictions.extend(predicted.cpu().numpy())
                
                if (epoch + 1) % epochs_to_save == 0:
                    
                    plot_confusion_matrix(
                         true_labels = np.array(validation_labels) + 1,
                         predicted_labels = np.array(validation_predictions) + 1
                    )

                    #Save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        },f"classification_models/checkpoint_{model_name}.pth")

            validation_accuracy.append(correct / total)

            print(f"""Epoch {epoch+1}, Training Loss: {training_losses[-1]},
                   Validation Loss: {validation_losses[-1]}
                   Accuracy: {validation_accuracy[-1]}%""")
            
            writer.add_scalar("Average training loss", np.mean(training_losses), epoch)
            writer.add_scalar("Average validation loss", np.mean(validation_losses), epoch)
            writer.add_scalar("Average accuracy", np.mean(validation_accuracy), epoch)

    writer.flush()