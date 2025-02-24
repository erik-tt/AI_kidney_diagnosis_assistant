from radiomics.featureextractor import RadiomicsFeatureExtractor
import numpy as np
import SimpleITK as sitk
from ML.segmentation.utils.file_reader import FileReader
import logging
from radiomics import logger
import pandas as pd
from tqdm import tqdm
import os

logger.setLevel(logging.ERROR)

def extract_radiomic_features(image_series, mask):
    image = sitk.ReadImage(image_series)  
    mask = sitk.ReadImage(mask)  

    image_array = sitk.GetArrayFromImage(image)  
    mask_array = sitk.GetArrayFromImage(mask) 
    mask_array = mask_array.squeeze()

    extractor = RadiomicsFeatureExtractor()
    extractor.loadParams("./settings.yaml")

    radiomic_features = {}

    for t in range(image_array.shape[0]):
        image_slice = sitk.GetImageFromArray(image_array[t])
        mask_slice = sitk.GetImageFromArray(mask_array)


        for region in np.unique(mask_array):
            if region == 0:
                continue
            
            if t not in radiomic_features:
                radiomic_features[t] = {}

            extractor.settings['label'] = region

            results = extractor.execute(image_slice, mask_slice)
            radiomic_features[t][region] = results


    return radiomic_features

def save_radiomic_features(data_dict):
    file_reader = FileReader("../../../../data/") 
    segmentation_data = []
    for path, suffices in data_dict.items():
        data_entry = file_reader.get_image_data(path, suffices) 
        print(data_entry)
        segmentation_data.extend(data_entry)

    data = []
    for entry in tqdm(segmentation_data):
        print(entry)
        image_series = entry["post"] # Ikke hardkode
        label = entry["label"]
        image_name = os.path.splitext(os.path.basename(image_series))[0]

        features = extract_radiomic_features(image_series, label)    

        for timestep, value in features.items():
            for region, features in value.items():
                row = {
                        "ImageName": image_name,
                        "TimeStep": int(timestep),
                        "Region": region,
                        **features
                    }
                data.append(row)
                
    df = pd.DataFrame(data)
    df.to_csv("../../../../data/radiomics_features.csv", index=False) 

data_dict = {
    "drsbru/post": ["label"],  # Folder: drsbru, Files: *_post.dcm, *_mask.nii.gz
    "drsprg/post": ["post"]
}

save_radiomic_features(data_dict=data_dict)