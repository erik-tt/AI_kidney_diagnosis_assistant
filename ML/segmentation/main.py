from argparse import ArgumentParser
from monai.data import DataLoader
from train_loops.train import train_loop, k_fold_validation
from config.model_selector import model_selector
from utils.create_dataset import create_dataset, create_dataset_kfold
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
import numpy as np
import os
#from torchinfo import summary

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use cuda: {torch.cuda.is_available()}")

    #Random seed locking (42 is the answer)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #Randomness in data loader
    g = torch.Generator()
    g.manual_seed(42)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("./runs", exist_ok=True)
    os.makedirs("./segmentation_models", exist_ok=True)
    log_dir = os.path.join("./runs", f"{params.model}_experiment_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)

    writer.add_text("Model", f"Model: {params.model}", global_step=0)
    writer.add_text("Transforms", f"Transforms: {params.transforms}", global_step=0)
    writer.add_text("Batch size", f"Batch size: {params.batch_size}", global_step=0)
    writer.add_text("Learning rate", f"Learning rate: {params.lr}", global_step=0)
    writer.add_text("Datadir", f"Data directories: {params.data_dir}", global_step=0)
    writer.add_text("Data suffix", f"Data directories: {params.data_suffix}", global_step=0)

    model = model_selector(params.model, device)
    
    #summary(model, input_size=(4, 1, 128, 128), verbose = 1, depth=40)
    
    
    if params.k_fold:
        data = create_dataset_kfold(params.data_dir, params.data_suffix)
        k_fold_validation(model_name=params.model,
                      dataset=data, 
                      epochs=params.num_epochs, 
                      batch_size=params.batch_size, 
                      device=device,
                      writer=writer,
                      transforms_name=params.transforms,
                      num_workers=params.num_workers,
                      splits=params.k_fold)
    
    else:
        train_dataset, test_dataset = create_dataset(params.data_dir, params.data_suffix, params.transforms)

        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, worker_init_fn=seed_worker(), generator=g)
        val_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False) 

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


    parser.add_argument("--data_dir", nargs='+', default=["drsprg", "drsbru"], help="Allowed data directories")
    parser.add_argument("--data_suffix", nargs='+', default=["POST"], help="Allowed suffices")
    parser.add_argument("--transforms", default="baseline")
    parser.add_argument("--model", default="unet")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--save",type=int, default=2)
    parser.add_argument("--k_fold",type=int, default = 10)
    args = parser.parse_args()

    main(args)

