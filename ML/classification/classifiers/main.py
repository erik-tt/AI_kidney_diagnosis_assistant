from argparse import ArgumentParser
from monai.data import DataLoader, pad_list_data_collate
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
    os.makedirs("./classification_models", exist_ok=True)
    log_dir = os.path.join("./runs", f"experiment_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)

    writer.add_text("Model", f"Model: {params.model}", global_step=0)
    writer.add_text("Transforms", f"Transforms: {params.transforms}", global_step=0)
    writer.add_text("Batch size", f"Batch size: {params.batch_size}", global_step=0)
    writer.add_text("Learning rate", f"Learning rate: {params.lr}", global_step=0)
    # writer.add_text("Datadir", f"Data directories: {params.data}", global_step=0)

    train_dataset, test_dataset = create_dataset(params.transforms, params.data_type)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=pad_list_data_collate)
    val_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, collate_fn=pad_list_data_collate) 

    model = model_selector(params.model, device)

    train_loop(
        model=model,
        epochs=params.num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        writer=writer,
        epochs_to_save=params.save,
        model_name=params.model
    )

if __name__ == "__main__":
    parser = ArgumentParser()

   # parser.add_argument("--data", nargs='+', default="", help="List of data directories") #Update this when we know dir structure for this data
    parser.add_argument("--model", default="resnet18")
    #image or time_series
    parser.add_argument("--data_type", default="image")
    #Set to config_2 if models have 3 input channels and image is the data-type, for example for pretrained resnet models
    parser.add_argument("--transforms", default="pretrained")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--save",type=int, default=2) #TODO:implement save

    args = parser.parse_args()

    main(args)
