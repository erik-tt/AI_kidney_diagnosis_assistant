from argparse import ArgumentParser
import random
from monai.data import DataLoader, pad_list_data_collate
import numpy as np
from train_loops.train import train_loop
from config.model_selector import model_selector
from utils.create_dataset import create_dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #For reproducability (The answer is 42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("./classification_models", exist_ok=True)

    base_log_dir = os.path.abspath("./runs")
    log_dir = os.path.join(base_log_dir, f"experiment_{timestamp}")
    os.makedirs(log_dir, exist_ok=True) 

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("Model", f"Model: {params.model}", global_step=0)
    writer.add_text("Transforms", f"Transforms: {params.transforms}", global_step=0)
    writer.add_text("Batch size", f"Batch size: {params.batch_size}", global_step=0)
    writer.add_text("Learning rate", f"Learning rate: {params.lr}", global_step=0)

    train_dataset, test_dataset = create_dataset(transforms_name=params.transforms, 
                                                    data_dir=params.data_dir, 
                                                    data_suffices=params.data_suffix, 
                                                    start_frame=params.start_frame,
                                                    end_frame=params.end_frame,
                                                    agg=params.agg,
                                                    cache=params.cache)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=pad_list_data_collate)
    val_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, collate_fn=pad_list_data_collate) 

    model = model_selector(params.model, device)
    for name, _ in model.named_modules(): print(name)

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
    
    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()

   # parser.add_argument("--data", nargs='+', default="", help="List of data directories") #Update this when we know dir structure for this data
    parser.add_argument("--model", default="resnet18")
    #image or time_series
    parser.add_argument("--data_dir", nargs='+', default=["drsbru", "drsprg"], help="Allowed data directories")
    parser.add_argument("--data_suffix", nargs='+', default=["POST"], help="Allowed suffices")
    #Set to config_2 if models have 3 input channels and image is the data-type, for example for pretrained resnet models
    parser.add_argument("--transforms", default="pretrained")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--save",type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--cache", type=bool, default=False)
    parser.add_argument("--agg", default="mean") # mean or time_series

    args = parser.parse_args()

    main(args)
