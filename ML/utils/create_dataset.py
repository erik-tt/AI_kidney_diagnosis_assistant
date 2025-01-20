from .file_reader import FileReader
from typing import List
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset

def create_dataset(paths: List[str],  
                   test_size: int = 0.2, 
                   random_state: int = 42, 
                   shuffle: bool = True): ## Add transforms
    
    file_reader = FileReader("../data")
    dataset = []
    for path in paths:
        segmentation_data = file_reader.get_segmentation_file_paths(path)
        dataset.append(segmentation_data)

    train_data, test_data = train_test_split(
        segmentation_data,
        test_size=test_size,      
        random_state=random_state,     
        shuffle=shuffle         
    )
    
    # Check if cache fails
    train_dataset = CacheDataset(train_data)
    test_dataset = CacheDataset(test_data)

    return train_dataset, test_dataset


