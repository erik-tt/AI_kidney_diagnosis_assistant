import pandas as pd
from typing import List, Optional
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
METADATA_PATH = os.path.join(BASE_DIR, "../../data/metadata.csv")  # Convert to absolute path

def get_segmentation_data(datasets: Optional[List[str]]=None, suffixes: Optional[List[str]]=None):  
    metadata = pd.read_csv(METADATA_PATH, usecols=["ImagePath", "SegLabelPath", "Suffix", "Database"])
    
    if datasets is not None:
        metadata = metadata[metadata["Database"].isin(datasets)]
    
    if suffixes is not None:
        metadata = metadata[metadata["Suffix"].isin(suffixes)]

    metadata = metadata[["ImagePath", "SegLabelPath"]]
    metadata.rename(columns={"ImagePath": "image", "SegLabelPath": "label"}, inplace=True)
    metadata_dict = metadata.to_dict(orient="records")

    return metadata_dict

def get_classification_data(datasets: Optional[List[str]]=None, suffixes: Optional[List[str]]=None):  
    metadata = pd.read_csv(METADATA_PATH, usecols=["TimeSeriesPath", "CKD", "Suffix", "Database"])
    
    if datasets is not None:
        metadata = metadata[metadata["Database"].isin(datasets)]
    
    if suffixes is not None:
        metadata = metadata[metadata["Suffix"].isin(suffixes)]


    metadata = metadata[["TimeSeriesPath", "CKD"]]
    metadata = metadata[["TimeSeriesPath", "CKD"]].dropna(subset=["CKD"])
    metadata["CKD"] = metadata["CKD"].astype(int)

    metadata.rename(columns={"TimeSeriesPath": "image", "CKD": "label"}, inplace=True)
    metadata_dict = metadata.to_dict(orient="records")

    return metadata_dict


def get_radiomic_data(datasets: Optional[List[str]]=None, suffixes: Optional[List[str]]=None):  
    metadata = pd.read_csv(METADATA_PATH, usecols=["TimeSeriesPath", "SegLabelPath", "Suffix", "Database"])
    
    if datasets is not None:
        metadata = metadata[metadata["Database"].isin(datasets)]
    
    if suffixes is not None:
        metadata = metadata[metadata["Suffix"].isin(suffixes)]

    metadata = metadata[["TimeSeriesPath", "SegLabelPath"]]

    metadata.rename(columns={"TimeSeriesPath": "image", "SegLabelPath": "label"}, inplace=True)
    metadata_dict = metadata.to_dict(orient="records")

    return metadata_dict

