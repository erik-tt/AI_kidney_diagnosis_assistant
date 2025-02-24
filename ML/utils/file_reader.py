import pandas as pd
from typing import List, Optional

METADATA_PATH = "../../data/metadata.csv"

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
