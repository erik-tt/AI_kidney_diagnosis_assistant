import torch
from monai.data import Dataset, CacheDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer

#Use dataset if it crashes
class ClassificationDataset(CacheDataset):
    def __init__(self, data_list, radiomics: bool, train: bool, start_frame=0, end_frame=None, agg="time_series", transforms=None):
        
        valid_agg_options = {"mean", "time_series"}
        if agg not in valid_agg_options:
            raise ValueError(f"Invalid agg='{agg}'. Must be one of {valid_agg_options}.")
        if start_frame < 0:
            raise ValueError(f"Invalid start_frame={start_frame}: Cannot be negative.")
        if end_frame is not None and end_frame <= start_frame:
            raise ValueError(f"Invalid end_frame={end_frame}: Must be greater than start_frame={start_frame}.")

        super().__init__(data=data_list, transform=transforms)

        self.start_frame = start_frame
        self.end_frame = end_frame
        self.agg = agg
        self.scaler = StandardScaler()
        self.radiomics = radiomics
        
        ## FIT SCALER AND IMPUTER TO TRAINING DATA
        if train and self.radiomics:
            all_features = []
            all_labels = []

            for entry in data_list:

                radiomics_path = entry["radiomics"]
                label = entry["label"]
                radiomic_data = np.load(radiomics_path, allow_pickle=True)
                
                features = radiomic_data["feature_values"]

                all_features.append(features)
                all_labels.append(label)

            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.array(all_labels)  

            valid_rows = ~np.isnan(all_features).any(axis=1)
            self.scaler.fit(all_features[valid_rows]) 

    def __getitem__(self, idx):
        # LOAD SAMPLE
        sample = super().__getitem__(idx)

        image = sample["image"]
        label = sample["label"]
        
        # 0-index label
        label = label - 1

        # LOAD, IMPUTE AND SCALE RADIOMIC FEATURES
        if self.radiomics:
            radiomics = np.load(sample["radiomics"], allow_pickle=True)
            feature_values = radiomics["feature_values"]

            feature_values = self.scaler.transform(feature_values)
            feature_values = np.nan_to_num(feature_values, nan=0.0)
            scaled_features = feature_values.squeeze(0)

        # NOISY LABEL CREATION
        sigma = 0.5  
        noisy_label = np.float32(np.random.normal(loc=float(label), scale=sigma))
        
        # AGGREGATE IMAGE
        if self.agg == "mean":
            total_frames = image.shape[1]

            end_frame = self.end_frame if self.end_frame is not None else total_frames
            start_frame = torch.clamp(torch.tensor(self.start_frame), 0, total_frames - 1).item()
            end_frame = torch.clamp(torch.tensor(self.end_frame if self.end_frame else total_frames), start_frame + 1, total_frames).item()

            image = image[:, start_frame:end_frame, :, :]

            image = image.mean(dim=1, keepdim=False) 
        
        # EMPTY TENSOR IF NOT RADIOMICS
        if not self.radiomics:
            scaled_features = torch.tensor([])

        return {"image": image, "label": label, "noisy_label": noisy_label, "radiomics": scaled_features}
    
    # RETURN IMPORT OBJECTS FOR VAL SET
    def get_scaler(self):
        return self.scaler
