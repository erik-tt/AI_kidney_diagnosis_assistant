from argparse import ArgumentParser
from monai.data import DataLoader
from train_loops.train import train_loop
from config.model_selector import model_selector
from utils.create_dataset import create_dataset
import torch
import os

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = create_dataset(params.transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers) 

    model = model_selector(params.model, device)

    train_loop(
        model=model,
        epochs=params.num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs_to_save=params.save,
        model_name=params.model
    )

if __name__ == "__main__":
    parser = ArgumentParser()

   # parser.add_argument("--data", nargs='+', default="", help="List of data directories") #Update this when we know dir structure for this data
    parser.add_argument("--transforms", default="default")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--save",type=int, default=10) #TODO:implement save

    args = parser.parse_args()

    main(args)
