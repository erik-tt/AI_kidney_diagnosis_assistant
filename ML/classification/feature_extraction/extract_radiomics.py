from radiomics.featureextractor import RadiomicsFeatureExtractor
import numpy as np
import SimpleITK as sitk
import logging
from radiomics import logger
import pandas as pd
from tqdm import tqdm
import os
import json
from ML.utils.file_reader import get_classification_data

logger.setLevel(logging.ERROR)

def extract_radiomic_features(image_series, mask):
    image = sitk.ReadImage(image_series)  
    mask = sitk.ReadImage(mask)  

    image_array = sitk.GetArrayFromImage(image)  
    mask_array = sitk.GetArrayFromImage(mask) 
    mask_array = mask_array.squeeze()

    time_extractor = RadiomicsFeatureExtractor()
    time_extractor.loadParams("./time_dependent_features.yaml")

    static_extractor = RadiomicsFeatureExtractor()
    static_extractor.loadParams("./time_independent_features.yaml")

    mask_array[(mask_array == 38) | (mask_array == 75)] = 1 

    time_feature_names = None
    time_features = []

    static_feature_names = None
    static_features = []

    for t in range(image_array.shape[0]):
        image_slice = sitk.GetImageFromArray(image_array[t])
        mask_slice = sitk.GetImageFromArray(mask_array)

        for region in np.unique(mask_array):
            if region == 0:
                continue ## kan kanskje ha med background?
            
            if t == 0:
                static_extractor.settings['label'] = region
                static_results = static_extractor.execute(image_slice, mask_slice)

                static_feature_names = list(static_results.keys())
                static_features = [static_results[feature].item() for feature in static_feature_names]  


            time_extractor.settings['label'] = region
            time_results = time_extractor.execute(image_slice, mask_slice)

            if time_feature_names is None:
                time_feature_names = list(time_results.keys())

            feature_values = [time_results[feature].item() for feature in time_feature_names]  
            time_features.append(feature_values)
    
    return np.array(time_features), time_feature_names, np.array(static_features).reshape(1,-1), static_feature_names
    

def save_radiomic_features(metadata):

    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image = row["TimeSeriesPath"]
        label = row["SegLabelPath"]

        directory_path = os.path.dirname(image)
        image_name = os.path.splitext(os.path.basename(image))[0]
        save_path = os.path.join(directory_path, f"{image_name}_radiomics.npy")

        time_features, time_names, static_features, static_names = extract_radiomic_features(image, label) 
        
        np.savez_compressed(
            save_path, 
            time_dependent=time_features, 
            time_feature_names=time_names,
            static_features=static_features, 
            static_feature_names=static_names
        )
        
        metadata.at[index, "RadiomicFeaturePath"] = save_path

    metadata.to_csv("../../../data/metadata.csv", index=False)

metadata = pd.read_csv("../../../data/metadata.csv") # Fix dette i pipeline
save_radiomic_features(metadata)

