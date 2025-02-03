import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_radiomic_features(path):
    
    radiomic_features = pd.read_csv(path, low_memory=False)

    radiomic_features["Suffix"] = radiomic_features["ImageName"].str.extract(r"([^_]+)$")
    radiomic_features["ImageName"] = radiomic_features["ImageName"].str.replace(r"(_[^_]+)$", "", regex=True)

    return radiomic_features

def process_label(path):
    label = pd.read_csv(path)
    label = label[["STUDY NAME", "CKD"]]
    label["STUDY NAME"] = label["STUDY NAME"].str.strip()
    label.dropna(inplace=True)

    label["CKD"] = label[["CKD"]].astype(int) - 1 # CKD stage start at 0
    
    return label

def process_df(radiomic_feature_path, label_paths, feature_columns = None):
    labels = [process_label(label_path) for label_path in label_paths]
    all_labels = pd.concat(labels, ignore_index=True)

    radiomic_features = process_radiomic_features(radiomic_feature_path)
    

    if not feature_columns:
        excluded_features = ["ImageName", "TimeStep", "Region", "Suffix"]
        feature_columns = [col for col in radiomic_features.columns if col not in excluded_features]

    df = pd.merge(radiomic_features, all_labels, how='left', left_on="ImageName", right_on="STUDY NAME")
    df.drop(columns=["STUDY NAME"], inplace = True)

    # FOR NÅ: drsprg og drsbru har forskjellig mengder bilder
    # Burde også sjekke om noen entries har nans
    df = df[df["ImageName"].str.contains("drsprg", case=False, na=False)]

    # FOR NÅ: 025 og 065 har ikke 180 bilder, sjekke om dette stemmer
    timestep_counts = df.groupby("ImageName")["TimeStep"].nunique()
    most_common_count = timestep_counts.mode()[0]

    image_names_to_remove = timestep_counts[timestep_counts != most_common_count].index.tolist() # 025 og 065 har ikke 180 bilder, sjekke om dette stemmer

    df = df[~df["ImageName"].isin(image_names_to_remove)]

    # Burdee egt concatenate region
    df_grouped = df.groupby(["ImageName", "Region", "CKD"])

    X = np.stack(df_grouped[feature_columns].apply(lambda x: x.to_numpy()).values) 
    y = df_grouped["CKD"].first().values
    
    # Scale each feature
    num_samples, num_timesteps, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)  

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    X_scaled = X_scaled.reshape(num_samples, num_timesteps, num_features) 
    
    # NB: ser ut som om det muligens er noe feil med region 75, nesten null på original_first_order_energy
    # Kan prøve å concatenate time dimensjonen med tsfresh
    return X_scaled, y
