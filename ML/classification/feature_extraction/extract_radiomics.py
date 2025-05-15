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
import os

logger.setLevel(logging.ERROR)

def extract_radiomic_features(image_series, mask, save_path):
    image = sitk.ReadImage(image_series)  
    mask = sitk.ReadImage(mask) 

    image_array = sitk.GetArrayFromImage(image)  
    mask_array = sitk.GetArrayFromImage(mask) 
    mask_array = mask_array.squeeze()

    time_path = os.path.join(os.path.dirname(__file__), "time_dependent_features.yaml")
    time_extractor = RadiomicsFeatureExtractor()
    time_extractor.loadParams(time_path)
    
    static_path = os.path.join(os.path.dirname(__file__), "time_independent_features.yaml")
    static_extractor = RadiomicsFeatureExtractor()
    static_extractor.loadParams(static_path)

    time_features_all = []
    static_features = []

    kidney_labels = [38, 75]

    mean_all = np.mean(image_array, axis=0)

    for region in kidney_labels:
        
        binary_mask_array = (mask_array == region).astype(np.uint8)
        binary_mask = sitk.GetImageFromArray(binary_mask_array)

        if np.sum(binary_mask_array) == 0:
            print("Missing Kidney")
            continue 

        mean_all_image = sitk.GetImageFromArray(mean_all)
        static_extractor.settings['label'] = 1
        static_results = static_extractor.execute(mean_all_image, binary_mask)

        static_feature_names = list(static_results.keys())
        static_values = [static_results[feature].item() for feature in static_feature_names]
        static_features.append([save_path, region] + static_values)

        for t in range(image_array.shape[0]):
            image_slice = sitk.GetImageFromArray(image_array[t])
            time_extractor.settings['label'] = 1
            time_results = time_extractor.execute(image_slice, binary_mask)

            if t == 0 and region == kidney_labels[0]:
                names = list(time_results.keys())

            feature_values = [time_results[feature].item() for feature in names]
            time_features_all.append([save_path, region, t] + feature_values)

    time_all_names = ["time_all_" + name for name in names]
    time_features_all_df = pd.DataFrame(time_features_all, columns=["id", "kidney_label", "timestep"] + time_all_names)
    static_features_df = pd.DataFrame(static_features, columns=["id", "kidney_label"] + static_feature_names)

    static_features_wide = static_features_df.pivot(index='id', columns='kidney_label')

    static_features_wide.columns = [
        f"{k}_{feature}" for k, feature in static_features_wide.columns
    ]

    static_features_wide = static_features_wide.reset_index()

    return time_features_all_df, static_features_wide

    
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

    tsfresh_per_kidney = []
    for kidney_region in [38, 75]:
        kidney_df = time_features_all_df[time_features_all_df["kidney_label"] == kidney_region].copy()
        
        extracted = extract_features(
            kidney_df.drop(columns=["kidney_label"]),
            column_id="id",
            column_sort="timestep",
            default_fc_parameters=MinimalFCParameters()
        )

        
        suffix = "_left" if kidney_region == 38 else "_right"
        extracted.columns = [f"{col}{suffix}" for col in extracted.columns]

        extracted = extracted.reset_index()
        print(extracted["index"])
        tsfresh_per_kidney.append(extracted)

    tsfresh_left, tsfresh_right = tsfresh_per_kidney
    extracted_features_all = pd.merge(tsfresh_left, tsfresh_right, on="index", how="outer")
    extracted_features_all = extracted_features_all.set_index("index")

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

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)  

    metadata_path = os.path.join(data_dir, "metadata.csv")
    metadata.to_csv(metadata_path, index=False)

## NB DETTE ER FRA GROUND TRUTH SEGMENTATION LABELS
