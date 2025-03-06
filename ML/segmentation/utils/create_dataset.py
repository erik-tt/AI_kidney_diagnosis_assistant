from typing import List
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset
from config.transforms_selector import transforms_selector
from torch.utils.tensorboard import SummaryWriter
from ML.utils.file_reader import get_segmentation_data

def create_dataset(datadirs: List[str],
                   suffices: List[str],
                   transforms_name: str,   
                   test_size: int = 0.2, 
                   random_state: int = 42, 
                   shuffle: bool = True
                   ):
    
    dataset = get_segmentation_data(datadirs, suffices)

    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,      
        random_state=random_state,     
        shuffle=shuffle,
    )
            
    train_transforms, val_transforms = transforms_selector(transforms_name)
    
    # Check if cache fails
    train_dataset = CacheDataset(train_data, train_transforms)
    test_dataset = CacheDataset(test_data, val_transforms)

    return train_dataset, test_dataset


def create_dataset_kfold(datadirs: List[str],
                         suffices: List[str]):
    
    return get_segmentation_data(datadirs, suffices)
    

