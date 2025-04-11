from radiomics.featureextractor import RadiomicsFeatureExtractor
import numpy as np
import SimpleITK as sitk
import logging
from radiomics import logger
import pandas as pd
from tqdm import tqdm
import os
from tsfresh import extract_features
import json
from ML.utils.file_reader import get_classification_data
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

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

    time_feature_names = None
    time_features_all = []
    time_features_select = []

    static_feature_names = None
    static_features = []

    time_extractor.settings['label'] = 1

    mask = sitk.GetImageFromArray(mask_array)

    mean_all = np.mean(image_array, axis=0)
    mean_all = sitk.GetImageFromArray(mean_all)

    mean_all_results = time_extractor.execute(mean_all, mask)
    names = list(mean_all_results.keys())

    mean_all_values = [mean_all_results[feature].item() for feature in names]  
    mean_all_features = [[save_path] + mean_all_values]
    mean_all_names = ["mean_all_" + name for name in names]



    mean_select = np.mean(image_array[6:19, :, :], axis=0)
    mean_select = sitk.GetImageFromArray(mean_select)

    mean_select_results = time_extractor.execute(mean_select, mask)
    mean_select_values = [mean_select_results[feature].item() for feature in names]  
    mean_select_features = [[save_path] + mean_select_values]
    mean_select_names = ["mean_select_" + name for name in names]


    for t in range(image_array.shape[0]):
        image_slice = sitk.GetImageFromArray(image_array[t])

        for region in np.unique(mask_array):
            if region == 0:
                continue ## kan kanskje ha med background?
            
            if t == 0:
                static_extractor.settings['label'] = region
                static_results = static_extractor.execute(image_slice, mask)

                static_feature_names = list(static_results.keys())
                static_values = [static_results[feature].item() for feature in static_feature_names]  
                static_features.append([save_path] + static_values)


            time_extractor.settings['label'] = region
            time_results = time_extractor.execute(image_slice, mask)

            #if time_feature_names is None:
            #    time_feature_names = list(time_results.keys())

            feature_values = [time_results[feature].item() for feature in names]  
            time_features_all.append([save_path, t] + feature_values)
            if t >=6 and t <= 18:
                time_features_select.append([save_path, t-6] + feature_values)
    
    time_select_names = ["time_select_" + name for name in names]
    time_all_names = ["time_all_" + name for name in names]

    time_features_all_df = pd.DataFrame(time_features_all, columns=["id", "timestep"] + time_all_names)
    time_features_select_df = pd.DataFrame(time_features_select, columns=["id", "timestep"] + time_select_names)
    mean_select_df = pd.DataFrame(mean_select_features, columns=["id"] + mean_select_names)
    mean_all_df = pd.DataFrame(mean_all_features, columns=["id"] + mean_all_names)
    static_features_df = pd.DataFrame(static_features, columns=["id"] + static_feature_names)

    return time_features_all_df, time_features_select_df, mean_all_df, mean_select_df, static_features_df
    
def save_radiomic_features(metadata):

    all_time_features = []
    select_time_features = []
    mean_all_features = []
    mean_select_features = []
    all_static_features = []

    metadata = metadata[metadata["Database"] == "drsbru"]

    metadata = metadata.iloc[60:]

    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image = row["TimeSeriesPath"]
        label = row["SegLabelPath"]

        directory_path = os.path.dirname(image)
        image_name = os.path.splitext(os.path.basename(image))[0]
        save_path = os.path.join(directory_path, f"{image_name}_radiomics.npz")

        time_features_all, time_features_select, mean_all, mean_select, static_features = extract_radiomic_features(image, label, save_path) 
        

        all_time_features.append(time_features_all)
        select_time_features.append(time_features_select)
        mean_all_features.append(mean_all)
        mean_select_features.append(mean_select)
        all_static_features.append(static_features)

    time_features_all_df = pd.concat(all_time_features, ignore_index=True)
    time_features_select_df = pd.concat(select_time_features, ignore_index=True)
    mean_all_features_df = pd.concat(mean_all_features, ignore_index=True)
    mean_select_features_df = pd.concat(mean_select_features, ignore_index=True)
    static_features_df = pd.concat(all_static_features, ignore_index=True)

    mean_all_features_df = mean_all_features_df.set_index("id")
    mean_select_features_df = mean_select_features_df.set_index("id")
    static_features_df = static_features_df.set_index("id")

    extracted_features_all = extract_features(time_features_all_df, column_id="id", column_sort="timestep", default_fc_parameters=ComprehensiveFCParameters())
    extracted_features_select = extract_features(time_features_select_df, column_id="id", column_sort="timestep", default_fc_parameters=ComprehensiveFCParameters())

    all_features = pd.concat([
        static_features_df, 
        extracted_features_all, 
        extracted_features_select, 
        mean_all_features_df, 
        mean_select_features_df
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

    #metadata.to_csv("../../../data/metadata.csv", index=False)

metadata = pd.read_csv("../../../data/metadata.csv") # Fix dette i pipeline
save_radiomic_features(metadata)

