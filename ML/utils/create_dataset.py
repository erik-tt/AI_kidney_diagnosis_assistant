from .file_reader import FileReader
from typing import List
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset
from config.transforms_selector import transforms_selector
from torch.utils.tensorboard import SummaryWriter

def create_dataset(paths: List[str],
                   transforms_name: str,   
                   test_size: int = 0.2, 
                   random_state: int = 42, 
                   shuffle: bool = True):
    
    file_reader = FileReader("../data")
    dataset = []
    for path in paths:
        segmentation_data = file_reader.get_segmentation_file_paths(path)
        dataset.extend(segmentation_data)

    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,      
        random_state=random_state,     
        shuffle=shuffle         
    )
    
    train_transforms, val_transforms = transforms_selector(transforms_name)
    
    # Check if cache fails
    train_dataset = CacheDataset(train_data, train_transforms)
    test_dataset = CacheDataset(test_data, val_transforms)

    return train_dataset, test_dataset


