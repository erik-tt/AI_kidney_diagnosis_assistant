from argparse import ArgumentParser
import random
from monai.data import DataLoader, pad_list_data_collate
from train_loops.train import train_loop, k_fold_validation
from config.model_selector import model_selector
from utils.create_dataset import create_dataset, create_dataset_kfold
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import random
import numpy as np
import monai
from ML.classification.classifiers.dataset.ClassificationDataset import ClassificationDataset

def set_seed(seed: int = 42):
    """Ensure reproducibility by setting all relevant seeds."""
    random.seed(seed)  # Python's built-in random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    monai.utils.set_determinism(seed=seed)  # MONAI-specific reproducibility
    torch.backends.cudnn.deterministic = True  # Ensure deterministic algorithms in cuDNN
    torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-tuning for deterministic behavior

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA Available:", torch.cuda.is_available())

    #For reproducability (The answer is 42)
    set_seed()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("./classification_models", exist_ok=True)

    base_log_dir = os.path.abspath("./runs")
    log_dir = os.path.join(base_log_dir, f"{params.model}_0.0001_{timestamp}")
    os.makedirs(log_dir, exist_ok=True) 

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("Model", f"Model: {params.model}", global_step=0)
    writer.add_text("Transforms", f"Transforms: {params.transforms}", global_step=0)
    writer.add_text("Batch size", f"Batch size: {params.batch_size}", global_step=0)
    writer.add_text("Learning rate", f"Learning rate: {params.lr}", global_step=0)


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
                      splits=10)
    else:
        model = model_selector(params.model, device)

        train_dataset, test_dataset = create_dataset(transforms_name=params.transforms, 
                                                        data_dir=params.data_dir, 
                                                        data_suffices=params.data_suffix, 
                                                        start_frame=params.start_frame,
                                                        end_frame=params.end_frame,
                                                        agg=params.agg,
                                                        radiomics=params.radiomics)

        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        val_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers) 


        train_loop(
            model=model,
            epochs=params.num_epochs,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            writer=writer,
            epochs_to_save=params.save,
            model_name=params.model,
            radiomics=params.radiomics
        )
        
        writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()

   # parser.add_argument("--data", nargs='+', default="", help="List of data directories") #Update this when we know dir structure for this data
    parser.add_argument("--model", default="resnet18")
    #image or time_series
    parser.add_argument("--data_dir", nargs='+', default=["drsprg"], help="Allowed data directories")
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
    parser.add_argument("--agg", default="mean") # mean or time_series
    parser.add_argument("--k_fold", type=bool, default=False)
    parser.add_argument("--radiomics", type=bool, default=False)

    args = parser.parse_args()

    main(args)
