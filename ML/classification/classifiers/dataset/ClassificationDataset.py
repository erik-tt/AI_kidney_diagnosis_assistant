import monai
import random
import torch
from monai.data import Dataset, CacheDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.impute import SimpleImputer
import pydicom

class ClassificationDataset(Dataset):
    def __init__(self, data_list, start_frame: int, end_frame: int, agg: str, cache: bool, radiomics: bool, train: bool, transforms=None):
        
        valid_agg_options = {"mean", "time_series"}
        if agg not in valid_agg_options:
            raise ValueError(f"Invalid agg='{agg}'. Must be one of {valid_agg_options}.")
        if start_frame < 0:
            raise ValueError(f"Invalid start_frame={start_frame}: Cannot be negative.")
        if end_frame is not None and end_frame <= start_frame:
            raise ValueError(f"Invalid end_frame={end_frame}: Must be greater than start_frame={start_frame}.")

        # FJERN DETTE
        self.data_list = []
        for entry in data_list:
            dicom_path = entry["image"]  # Path to the DICOM file
            try:
                dicom_data = pydicom.dcmread(dicom_path)
                num_frames = dicom_data.pixel_array.shape[0]  # Get number of frames
                
                if num_frames == 180 or num_frames == 120:  # Keep only images with exactly 180 frames
                    self.data_list.append(entry)
            except Exception as e:
                print(f"Skipping {dicom_path} due to error: {e}")

        super().__init__(data=self.data_list, transform=transforms)

        self.start_frame = start_frame
        self.end_frame = end_frame
        self.agg = agg
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.top_indices = None
        self.nan_cols = None
        self.radiomics = radiomics
        
        if train and self.radiomics:
            all_features = []
            all_labels = []

            for entry in data_list:
                radiomics_path = entry["radiomics"]
                label = entry["label"]
                radiomic_data = np.load(radiomics_path)
                
                features = radiomic_data["feature_values"]

                all_features.append(features)
                all_labels.append(label)

            all_features = np.concatenate(all_features, axis=0)
            all_labels = np.array(all_labels)   

            all_features[np.isinf(all_features)] = np.nan
            self.nan_cols = np.all(np.isnan(all_features), axis=0)
            all_features = all_features[:, ~self.nan_cols]

            all_features = self.imputer.fit_transform(all_features)

            correlations = np.array([np.corrcoef(all_features[:, i], all_labels)[0, 1] for i in range(all_features.shape[1])])

            self.top_indices = np.argsort(np.abs(correlations))[-201:]
            all_features = all_features[:, self.top_indices]

            self.scaler.fit(all_features)

            # Delete variables after fitting
        #if cache:   
            #self.dataset = CacheDataset(data=data_list, transform=transforms)
        #else:
            #self.dataset = Dataset(data=data_list, transform=transforms)

    def __len__(self):
        #return len(self.dataset) 
        return super().__len__() #can remove

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        #sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]
        if self.radiomics:
            radiomics = np.load(sample["radiomics"])
            feature_values = radiomics["feature_values"]

            feature_values[np.isinf(feature_values)] = np.nan

            feature_values = feature_values[:, ~self.nan_cols]
            feature_values = self.imputer.transform(feature_values)

            feature_values = feature_values[:, self.top_indices]

            self.scaler.transform(feature_values)
            scaled_features = feature_values.squeeze(0)

        sigma = 0.5  # Standard deviation, adjust as needed
        noisy_label = np.float32(np.random.normal(loc=float(label), scale=sigma))
        
        total_frames = image.shape[1]

        end_frame = self.end_frame if self.end_frame is not None else total_frames
        start_frame = torch.clamp(torch.tensor(self.start_frame), 0, total_frames - 1).item()
        end_frame = torch.clamp(torch.tensor(self.end_frame if self.end_frame else total_frames), start_frame + 1, total_frames).item()

        image = image[:, start_frame:end_frame, :, :]

        image = image.mean(dim=1, keepdim=False) if self.agg == "mean" else image
        if not self.radiomics:
            scaled_features = torch.tensor([])
        return {"image": image, "label": label, "noisy_label": noisy_label, "radiomics": radiomics}
    
    def get_objects(self):
        return self.scaler, self.imputer, self.top_indices, self.nan_cols
