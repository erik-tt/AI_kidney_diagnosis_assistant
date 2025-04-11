from typing import List
from sklearn.model_selection import train_test_split
from config.transforms_selector import transforms_selector
from dataset.SampleDataset import SampleDataset2
from monai.data import CacheDataset, Dataset
from monai.data import CacheDataset
from ML.utils.file_reader import get_classification_data
from ML.classification.classifiers.dataset.ClassificationDataset import ClassificationDataset

# REmove
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from autogluon.tabular import TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from ML.utils.file_reader import get_classification_data
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pydicom
from tsfresh import select_features
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox, yeojohnson
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from ML.utils.file_reader import get_classification_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from torchvision import models, transforms
from sklearn.ensemble import StackingClassifier
from PIL import Image
import torch
import os
from sklearn.impute import SimpleImputer
from monai.networks.nets import ResNetFeatures


def safe_log(x):
    return np.log1p(np.maximum(x, 0))  # log1p(x) = log(1 + x) to avoid log(0) issues

def safe_inverse(x):
    return np.where(x != 0, 1 / x, 0)  # Avoid division by zero

def safe_exp(x):
    return np.exp(x)  # Handles large values but may explode

def safe_square(x):
    return np.square(x)

# **Safe transformation functions with clipping**
def log_transform(x): return np.log1p(x)
def sqrt_transform(x): return np.sqrt(x)
def inverse_transform(x): return 1 / (x + 1e-8)
def exp_transform(x): return np.exp(np.clip(x, -50, 50))  # **Clip to avoid overflow**
def square_transform(x): return np.power(x, 2)
def cube_transform(x): return np.power(x, 3)
def recip_sqrt_transform(x): return 1 / np.sqrt(x + 1e-8)

def boxcox_transform(x): 
    if np.all(x > 0) and x.nunique() > 1:
        return np.clip(boxcox(x + 1e-8)[0], -1e10, 1e10)  # **Clip extreme values**
    return x

def yeojohnson_transform(x): 
    if x.nunique() > 1:
        return np.clip(yeojohnson(x + 1e-8)[0], -1e10, 1e10)  # **Clip extreme values**
    return x

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Lambda(lambXda x: x.repeat(3, 1, 1)),  # Convert grayscale to 3-channel
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet stats
    transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension (shape: [1, 3, H, W])
])

from monai.transforms import Resize, EnsureChannelFirst, ToTensor, NormalizeIntensity

preprocess2 = transforms.Compose([
    EnsureChannelFirst(channel_dim="no_channel"),  # Ensures (C, D, H, W) format
    Resize((180, 224, 224)),  # Use MONAI Resize, which supports 3D
    ToTensor(dtype=torch.float32),  # Convert to tensor
    NormalizeIntensity(),
    transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension (shape: [1, 3, H, W])
])

from scipy.stats import skew, kurtosis

BASEPATH = "/cluster/home/malovhoi/AI_kidney_diagnosis_assistant/data/BAZA dynamicrenal/drsprg/DATA_DICOM"

def create_dataset(test_size: float = 0.2, random_state: int = 42, shuffle: bool = True):
    """ Loads radiomics dataset, applies transformations & scaling BEFORE splitting, then trains an SVM model. """

    feature_extractor = models.resnet18(pretrained=True)
    #feature_extractor2 = ResNetFeatures("resnet50", pretrained=True)

    # Remove the final fully connected layer
    feature_extractor.fc = torch.nn.Identity()  # Replace fc layer with an identity layer

    # Set the model to evaluation mode
    feature_extractor.eval()
    ### **🚀 Step 1: Load Dataset**
    dataset = get_classification_data(["drsbru"], ["POST"], radiomics=True)

    all_features, all_labels, all_images = [], [], []
    names = None  # Store feature names

    j = -1
    all_f = []
    # **Extract Features & Labels**
    for entry in dataset:
        j += 1
        label = entry["label"]
        print(j)
        radiomics_path = entry["radiomics"]
        image_path = entry["image"]
        
        if pd.isna(label):
            print("skip")
            continue  # Move to the next entry
        
        #folder_name = os.path.basename(os.path.dirname(image_path))
    
        # Split and extract the drsprg_XXX part
        #drsprg_id = "_".join(folder_name.split("_")[:2])
        #path_post2 = f"{BASEPATH}/{drsprg_id}/{drsprg_id}_ANT.dcm"

        radiomic_data = np.load(radiomics_path, allow_pickle=True)

        features = radiomic_data["feature_values"].flatten()
        
        if names is None:  # Store feature names once
            names = radiomic_data["feature_names"]

        dicom_data = pydicom.dcmread(image_path)
        pixel_array = dicom_data.pixel_array  # Shape: (frames, height, width) or (height, width)

        counter = pixel_array.shape[0]
        image_features_list = []
        
        #print(j)
        if counter != 180 and counter != 120:
            continue
        
        #pixel_array = pixel_array.astype(np.float32)  # Convert <u2 to float32
        #print(pixel_array.shape)
        
        #mean_image_pil = Image.fromarray(image_tensor)
        #image_tensor = preprocess2(pixel_array)
        #image_tensor = image_tensor.to(torch.float32)  # Ensure tensor is float

        #with torch.no_grad():  # Disable gradient computation
            #image_features = feature_extractor2(image_tensor)# Shape: (512,)
            #image_features = image_features[-1]  # Use the final ResNet layer

            #extracted_features = image_features.flatten().cpu().numpy()  # Flatten for classifier
            #print("Extracted feature shape:", extracted_features.shape)

            #if extracted_features is not None:
            #    all_features.append(extracted_features)
            #    all_labels.append(label)
       # if image_features is not None:
            #image_features_list.extend(image_features)
        
        
        for i in range(counter):
            try:
                pix = pixel_array[i, :, :].astype(np.float32)
                
                #mean_image = np.mean(pixel_array, axis=0)  # Shape: (height, width)
                mean_image_pil = Image.fromarray(pix)

                image_tensor = preprocess(mean_image_pil)
                image_tensor = image_tensor.to(torch.float32)  # Ensure tensor is float

                with torch.no_grad():  # Disable gradient computation
                    image_features = feature_extractor(image_tensor).squeeze().numpy()  # Shape: (512,)
                if image_features is not None:
                    image_features_list.append(image_features)
                else:
                    print(image_path)
            except Exception as e:
                print(f"An error occurred: {e}")
                image_features = None  # Default to None or handle it appropriately

        time_series_features = np.array(image_features_list)
        all_f.append(time_series_features)
        print(time_series_features.shape)
        mean_features = np.mean(time_series_features, axis=0)  # Mean of each feature
        std_features = np.std(time_series_features, axis=0)  # Standard deviation
        skew_features = skew(time_series_features, axis=0)  # Skewness
        kurtosis_features = kurtosis(time_series_features, axis=0)  # Kurtosis

        # Stack into a single feature vector
        final_features = np.hstack([mean_features, std_features, skew_features, kurtosis_features])
        #combined_features = np.concatenate([features, image_features])

        # Append to lists # ONLY IMAGE FEATURES
        if image_features_list is not None:
            all_features.append(final_features.tolist())
            all_labels.append(label)

    image_feature_names = [f"image_feature_{i}" for i in range(final_features.shape[0])]

    all_f = np.array(all_f)
    all_labels = np.array(all_labels, dtype=int)  # Ensure numeric
    print(all_f.shape)
    np.save("features_drsbru.npy", all_f)  # Saves feature matrix
    np.save("labels_drsbru.npy", all_labels)  # Saves labels

    print(all_f.shape)
    # Combine radiomics feature names and image feature names
    all_feature_names = np.concatenate([names, image_feature_names])
    # Convert to DataFrame
    full_df = pd.DataFrame(all_features, columns=image_feature_names)

    # Drop columns where all values are NaN
    full_df = full_df.dropna(axis=1, how="all")

    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df = full_df.clip(lower=-1e12, upper=1e12)

    imputer = SimpleImputer(strategy="mean")  # Change to "median" or "most_frequent" if needed
    full_df[:] = imputer.fit_transform(full_df)

    print(f"all_labels length: {len(all_labels)}")
    print(f"full_df shape before adding CKD_stage: {full_df.shape}")
    # Efficiently combine transformed features with the original dataset
    all_labels = np.array(all_labels, dtype=int)  # Ensure numeric
    print(all_labels)
    full_df["CKD_stage"] = all_labels - 1 # Add CKD labels
    print(full_df["CKD_stage"])

    #Binary
    #full_df["CKD_stage"] = np.where(np.array(all_labels) <= 3, 0, 1)


    #full_df = select_features(full_df, np.array(all_labels))
    print(full_df.shape)
    #nan_columns = full_df.columns[full_df.isna().any()]  # Select columns with NaNs

    #full_df.dropna(axis=1, inplace=True)  # Drops columns with NaNs

    correlations = full_df.drop(columns=["CKD_stage"]).apply(lambda col: col.corr(full_df["CKD_stage"]))
    sorted_correlation = correlations.abs().sort_values(ascending=False)

    top_features = sorted_correlation.index[:200].tolist()  # Select top 10 features

    # Compute the correlation matrix for the selected features
    corr_matrix = full_df[top_features].corr().abs()

    # Define threshold for high correlation (e.g., 0.9)
    high_corr_threshold = 0.90

    # Find columns that are highly correlated with each other
    to_remove = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > high_corr_threshold:
                to_remove.add(corr_matrix.columns[i])

    # Keep only features that are not redundant
    selected_features = [feat for feat in top_features if feat not in to_remove]


    print("📊 Top 10 Features and Their Correlation with CKD_stage:")
    for feature in selected_features:
        print(f"{feature}: {correlations[feature]:.4f}")  # Print correlation values
    
    #X = full_df[selected_features]
    X = full_df.drop(columns=["CKD_stage"])
    y = full_df["CKD_stage"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 SVM Accuracy: {accuracy:.4f}")

    # Print classification report (Precision, Recall, F1-score)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 RF Accuracy: {accuracy:.4f}")

    # Convert to Pandas DataFrame and Series before resetting index
    X_train_pd = pd.DataFrame(X_train, columns=X.columns)  # Convert NumPy array to DataFrame
    y_train_pd = pd.Series(y_train, name="CKD_stage")  # Convert NumPy array to Series

    X_test_pd = pd.DataFrame(X_test, columns=X.columns)
    y_test_pd = pd.Series(y_test, name="CKD_stage")

    # Reset indices to ensure alignment
    X_train_pd = X_train_pd.reset_index(drop=True)
    y_train_pd = y_train_pd.reset_index(drop=True)
    X_test_pd = X_test_pd.reset_index(drop=True)
    y_test_pd = y_test_pd.reset_index(drop=True)

    # Now concatenate safely
    train_data = pd.concat([X_train_pd, y_train_pd], axis=1)
    test_data = pd.concat([X_test_pd, y_test_pd], axis=1)

    # Verify alignment
    print("Train Data After Fix:")
    print(train_data.head())

    predictor = TabularPredictor(label="CKD_stage", problem_type="multiclass").fit(train_data)

    # Evaluate AutoGluon
    test_acc = predictor.evaluate(test_data)
    print("AutoGluon Test Accuracy:", test_acc)
    
        # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test)

    # LightGBM parameters
    params = {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1
    }

    # Train LightGBM
    lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test])

    # Predict
    y_pred_lgb = lgb_model.predict(X_test)
    y_pred_lgb_labels = np.argmax(y_pred_lgb, axis=1)  # Convert probabilities to class labels

    # Compute accuracy
    lgb_acc = accuracy_score(y_test, y_pred_lgb_labels)
    print("LightGBM Test Accuracy:", lgb_acc)

    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    y_pred = logreg_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 LogReg Accuracy: {accuracy:.4f}")


    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Bayes Accuracy: {accuracy:.4f}")

    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 XGB Accuracy: {accuracy:.4f}")

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 KNN Accuracy: {accuracy:.4f}")

    # Base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42))
    ]

    # Meta-model (learns from base models' outputs)
    stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

    # Train the model
    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Ensemble Accuracy: {accuracy:.4f}")


    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_model.fit(X_train, y_train)
    y_pred = et_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 ExtraTreesClassifier Accuracy: {accuracy:.4f}")


    xgb_model = XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=100)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 XGB CrossEntropy Accuracy: {accuracy:.4f}")

    xgb_model = XGBClassifier(
        n_estimators=200,  # More trees but with early stopping
        learning_rate=0.05,  # Reduce step size
        max_depth=4,  # Prevent overfitting (keep it shallow)
        min_child_weight=2,  # Avoid splitting small nodes
        subsample=0.8,  # Randomly drop 20% of data per tree
        colsample_bytree=0.8,  # Use only 80% of features per tree
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 XGB Rgularization Accuracy: {accuracy:.4f}")

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization (adds sparsity)
        reg_lambda=1.0,  # L2 regularization (reduces model complexity)
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 XGB More Rgularization Accuracy: {accuracy:.4f}")


# Run dataset creation and model training
create_dataset()
