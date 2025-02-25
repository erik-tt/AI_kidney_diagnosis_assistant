from radiomics.featureextractor import RadiomicsFeatureExtractor
import numpy as np
import SimpleITK as sitk
from ML.utils.file_reader import get_radiomic_data
import logging
from radiomics import logger
import pandas as pd
from tqdm import tqdm
import os
import json

logger.setLevel(logging.ERROR)

def extract_radiomic_features(image_series, mask):
    image = sitk.ReadImage(image_series)  
    mask = sitk.ReadImage(mask)  

    image_array = sitk.GetArrayFromImage(image)  
    mask_array = sitk.GetArrayFromImage(mask) 
    mask_array = mask_array.squeeze()

    extractor = RadiomicsFeatureExtractor()
    extractor.loadParams("./settings.yaml")

    mask_array[(mask_array == 38) | (mask_array == 75)] = 1 

    radiomic_features = {}

    for t in range(image_array.shape[0]):
        image_slice = sitk.GetImageFromArray(image_array[t])
        mask_slice = sitk.GetImageFromArray(mask_array)

        for region in np.unique(mask_array):
            if region == 0:
                continue ## kan kanskje ha med background?
            
            extractor.settings['label'] = region
            results = extractor.execute(image_slice, mask_slice)

            if region not in radiomic_features:
                radiomic_features[region] = {feature_name: [] for feature_name in results.keys()}

            for feature_name, feature_value in results.items():
                radiomic_features[region][feature_name].append(feature_value.item())

    return radiomic_features

def save_radiomic_features(metadata):

    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image = row["TimeSeriesPath"]
        label = row["SegLabelPath"]

        directory_path = os.path.dirname(image)
        image_name = os.path.splitext(os.path.basename(image))[0]
        save_path = os.path.join(directory_path, f"{image_name}_radiomics.npy")

        radiomic_features = extract_radiomic_features(image, label) 

        np.save(save_path, radiomic_features)
        metadata.at[index, "RadiomicFeaturePath"] = save_path

    metadata.to_csv("../../../data/metadata.csv", index=False)

metadata = pd.read_csv("../../../data/metadata.csv") # Fix dette i pipeline
save_radiomic_features(metadata)
