from argparse import ArgumentParser
from monai.data import DataLoader
from train_loops.train import train_loop
from config.model_selector import model_selector
from utils.create_dataset import create_dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("./runs", exist_ok=True)
    log_dir = os.path.join("./runs", f"experiment_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)

    writer.add_text("Model", f"Model: {params.model}", global_step=0)
    writer.add_text("Transforms", f"Transforms: {params.transforms}", global_step=0)
    writer.add_text("Batch size", f"Batch size: {params.batch_size}", global_step=0)
    writer.add_text("Learning rate", f"Learning rate: {params.lr}", global_step=0)
    writer.add_text("Datadir", f"Data directories: {params.data}", global_step=0)

    train_dataset, test_dataset = create_dataset(params.data, params.transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers) 

    model = model_selector(params.model, device)

    train_loop(
        model=model,
        epochs=params.num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        writer=writer
    )

    writer.close()
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data", nargs='+', default=["drsprg/post", "drsbru/post"], help="List of data directories")
    parser.add_argument("--transforms", default="default")
    parser.add_argument("--model", default="UNet")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=int, default=0.001)

    args = parser.parse_args()

    main(args)

