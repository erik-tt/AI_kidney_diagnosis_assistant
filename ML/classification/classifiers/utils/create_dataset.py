from typing import List
from sklearn.model_selection import train_test_split
from config.transforms_selector import transforms_selector
from utils.file_reader import FileReader
from dataset.SampleDataset import SampleDataset2
from monai.data import CacheDataset, Dataset

def create_dataset(transforms_name: str,
                   data_type: str,   
                   test_size: int = 0.2, 
                   random_state: int = 42, 
                   shuffle: bool = True):
    
    file_reader = FileReader("../../../data", data_type=data_type)
    dataset = file_reader.get_classification_data()

    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,      
        random_state=random_state,     
        shuffle=shuffle 
    )
    train_transforms, val_transforms = transforms_selector(transforms_name)

    #train data and test data determinstic
    
    if data_type == "image":
        train_dataset = Dataset(train_data, train_transforms)
        test_dataset = Dataset(test_data, val_transforms)
    else:
        train_dataset = SampleDataset2(train_data, num_frames=20, transforms=train_transforms, agg="mean")
        test_dataset = SampleDataset2(test_data, num_frames=20, transforms=val_transforms, agg="mean")


    sample = train_dataset[0]["image"]
    print(f"Loaded image shape: {sample.shape}")

    return train_dataset, test_dataset


    

