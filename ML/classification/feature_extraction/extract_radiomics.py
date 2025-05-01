from radiomics.featureextractor import RadiomicsFeatureExtractor
import numpy as np
import SimpleITK as sitk
import logging
from radiomics import logger
import pandas as pd
from tqdm import tqdm
import os
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

logger.setLevel(logging.ERROR)

def extract_radiomic_features(image_series, mask, save_path):
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

    time_features_all = []

    static_feature_names = None
    static_features = []

    time_extractor.settings['label'] = 1

    mask = sitk.GetImageFromArray(mask_array)

    mean_all = np.mean(image_array, axis=0)
    mean_all = sitk.GetImageFromArray(mean_all)

    mean_all_results = time_extractor.execute(mean_all, mask)
    names = list(mean_all_results.keys()) # FIXE NAMES


    for t in range(image_array.shape[0]):
        image_slice = sitk.GetImageFromArray(image_array[t])
        
        for region in np.unique(mask_array):
            if region == 0:
                continue 
            
            if t == 0:
                static_extractor.settings['label'] = region
                static_results = static_extractor.execute(image_slice, mask)

                static_feature_names = list(static_results.keys())
                static_values = [static_results[feature].item() for feature in static_feature_names]  
                static_features.append([save_path] + static_values)


            time_extractor.settings['label'] = region
            time_results = time_extractor.execute(image_slice, mask)


            feature_values = [time_results[feature].item() for feature in names]  
            time_features_all.append([save_path, t] + feature_values)
        
    time_all_names = ["time_all_" + name for name in names]

    time_features_all_df = pd.DataFrame(time_features_all, columns=["id", "timestep"] + time_all_names)
    static_features_df = pd.DataFrame(static_features, columns=["id"] + static_feature_names)

    return time_features_all_df, static_features_df
    
def save_radiomic_features(metadata):

    all_time_features = []
    all_static_features = []

    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image = row["TimeSeriesPath"]
        label = row["SegLabelPath"]

        directory_path = os.path.dirname(image)
        image_name = os.path.splitext(os.path.basename(image))[0]
        save_path = os.path.join(directory_path, f"{image_name}_radiomics.npz")

        time_features_all,  static_features = extract_radiomic_features(image, label, save_path) 
        

        all_time_features.append(time_features_all)
        all_static_features.append(static_features)

    time_features_all_df = pd.concat(all_time_features, ignore_index=True)
    static_features_df = pd.concat(all_static_features, ignore_index=True)

    static_features_df = static_features_df.set_index("id")

    extracted_features_all = extract_features(time_features_all_df, column_id="id", column_sort="timestep", default_fc_parameters=MinimalFCParameters())
    extracted_features_all.index = extracted_features_all.index.get_level_values(0)

    all_features = pd.concat([
        static_features_df, 
        extracted_features_all, 
    ], axis=1)


    feature_names = all_features.columns
    for index, row in all_features.iterrows():
        file_path = row.name   
        metadata_id = os.path.basename(os.path.dirname(file_path))
        feature_values = row.to_numpy().reshape(1,-1)

        np.savez_compressed(
            file_path, 
            feature_values=feature_values, 
            feature_names=feature_names
        )

        metadata.loc[metadata["ImageName"] == metadata_id, "RadiomicFeaturePath"] = file_path

    metadata.to_csv("../../../data/metadata.csv", index=False)

if __name__ == "__main__":
    metadata = pd.read_csv("../../../data/metadata.csv")  
    save_radiomic_features(metadata)

## NB DETTE ER FRA GROUND TRUTH SEGMENTATION LABELS