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
                   cache: bool,
                   test_size: int = 0.2, 
                   random_state: int = 42, 
                   shuffle: bool = True):
    
    dataset = get_classification_data(data_dir, data_suffices, radiomics=False)

    #scale_radiomics(dataset)
    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,      
        random_state=random_state,     
        shuffle=shuffle 
    )

    train_transforms, val_transforms = transforms_selector(transforms_name) 
    
    # Check if cache fails
    train_dataset = ClassificationDataset(data_list=train_data, 
                                            start_frame=start_frame, 
                                            end_frame=end_frame,
                                            agg=agg, 
                                            cache=cache,
                                            transforms=train_transforms,
                                            radiomics=False,
                                            train=True
                                            )
    

    test_dataset = ClassificationDataset(data_list=test_data, 
                                            start_frame=start_frame, 
                                            end_frame=end_frame, 
                                            agg=agg, 
                                            cache=cache,
                                            transforms=val_transforms, 
                                            radiomics=False,
                                            train=False
                                            )

    test_dataset.scaler, test_dataset.imputer, test_dataset.top_indices, test_dataset.nan_cols = train_dataset.get_objects()

    sample = train_dataset[0]["image"]
    print(f"Loaded image shape: {sample.shape}")

    return train_dataset, test_dataset

def create_dataset_kfold(data_dir: List[str],
                            data_suffices: List[str]):
    
    return get_classification_data(data_dir, data_suffices, radiomics=True)
    