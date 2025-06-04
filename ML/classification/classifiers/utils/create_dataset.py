from typing import List
from sklearn.model_selection import train_test_split
from config.transforms_selector import transforms_selector
from monai.data import CacheDataset, Dataset
from ML.utils.file_reader import get_classification_data
from ML.classification.classifiers.dataset.ClassificationDataset import ClassificationDataset

def create_dataset(transforms_name: str,
                   data_dir: List[str],
                   data_suffices: List[str],   
                   start_frame: int,
                   end_frame: int, 
                   agg: str,
                   radiomics: bool,
                   test_size: int = 0.2, 
                   random_state: int = 42, # FIX
                   shuffle: bool = True):
    
    dataset = get_classification_data(data_dir, data_suffices, radiomics=radiomics)

    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,      
        random_state=random_state,     
        shuffle=shuffle 
    )

    train_transforms, val_transforms = transforms_selector(transforms_name) 
    
    train_dataset = ClassificationDataset(data_list=train_data, 
                                            start_frame=start_frame, 
                                            end_frame=end_frame,
                                            agg=agg, 
                                            transforms=train_transforms,
                                            radiomics=radiomics,
                                            train=True
                                            )
    

    test_dataset = ClassificationDataset(data_list=test_data, 
                                            start_frame=start_frame, 
                                            end_frame=end_frame, 
                                            agg=agg, 
                                            transforms=val_transforms, 
                                            radiomics=radiomics,
                                            train=False
                                            )

    test_dataset.scaler = train_dataset.get_scaler()

    sample = train_dataset[0]["image"]
    print(f"Loaded image shape: {sample.shape}")

    return train_dataset, test_dataset

def create_dataset_kfold(data_dir: List[str],
                            data_suffices: List[str]):
    
    return get_classification_data(data_dir, data_suffices, radiomics=True)
    