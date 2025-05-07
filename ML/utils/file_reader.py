import pandas as pd
from typing import List, Optional
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
METADATA_PATH = os.path.join(BASE_DIR, "../../data/metadata.csv")  # Convert to absolute path

def get_segmentation_data(
    datasets: Optional[List[str]]=None, 
    suffixes: Optional[List[str]]=None
    ):  

    metadata = pd.read_csv(METADATA_PATH, usecols=["ImagePath", "SegLabelPath", "Suffix", "Database"])
    
    if datasets is not None:
        metadata = metadata[metadata["Database"].isin(datasets)]
    
    if suffixes is not None:
        metadata = metadata[metadata["Suffix"].isin(suffixes)]

    metadata = metadata[["ImagePath", "SegLabelPath"]]
    metadata.rename(columns={"ImagePath": "image", "SegLabelPath": "label"}, inplace=True)
    metadata_dict = metadata.to_dict(orient="records")

    return metadata_dict

def get_classification_data(
    datasets: Optional[List[str]]=None, 
    suffixes: Optional[List[str]]=None, 
    radiomics: Optional[bool]=False
    ):  
    
    cols = ["TimeSeriesPath", "CKD"]

    if radiomics:
        cols.append("RadiomicFeaturePath")

    metadata = pd.read_csv(METADATA_PATH, usecols=cols + ["Suffix", "Database"])
    
    if datasets is not None:
        metadata = metadata[metadata["Database"].isin(datasets)]
    
    if suffixes is not None:
        metadata = metadata[metadata["Suffix"].isin(suffixes)]


    metadata = metadata[cols]
    metadata = metadata[cols].dropna(subset=["CKD"])
    metadata["CKD"] = metadata["CKD"].astype(int)

    rename_dict = {
        "TimeSeriesPath": "image",
        "CKD": "label"
    }
    
    if radiomics:
        rename_dict["RadiomicFeaturePath"] = "radiomics"

    metadata.rename(columns=rename_dict, inplace=True)
    
    metadata_dict = metadata.to_dict(orient="records")

    return metadata_dict


