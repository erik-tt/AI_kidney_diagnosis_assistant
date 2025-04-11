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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from ML.utils.file_reader import get_classification_data
import pydicom
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
from scipy.fftpack import fft
from sklearn.feature_selection import SequentialFeatureSelector



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
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to 3-channel
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
def hjorth_parameters(x):
    activity = np.var(x, axis=1)
    mobility = np.std(np.diff(x, axis=1), axis=1) / np.std(x, axis=1)
    complexity = np.std(np.diff(np.diff(x, axis=1), axis=1), axis=1) / mobility
    return np.hstack([activity, mobility, complexity])

BASEPATH = "/cluster/home/malovhoi/AI_kidney_diagnosis_assistant/data/BAZA dynamicrenal/drsprg/DATA_DICOM"

def create_dataset(test_size: float = 0.2, random_state: int = 42, shuffle: bool = True):
    """ Loads radiomics dataset, applies transformations & scaling BEFORE splitting, then trains an SVM model. """

    drsprg_features = np.load("features_drsprg.npy")
    drsprg_labels = np.load("labels_drsprg.npy")

    unique_labels, counts = np.unique(drsprg_labels, return_counts=True)

    # Print label counts
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} occurrences")

    print()
    drsprg_labels = drsprg_labels - 1
    # Keep only features that are not redundant
    drsprg_flat = drsprg_features.reshape(drsprg_features.shape[0], -1)
    print(drsprg_features.shape)
    mean_features = np.mean(drsprg_features, axis=1)  # Mean of each feature
    std_features = np.std(drsprg_features, axis=1)  # Standard deviation
    skew_features = skew(drsprg_features, axis=1)  # Skewness
    kurtosis_features = kurtosis(drsprg_features, axis=1)  # Kurtosis
    min_features = np.min(drsprg_features, axis=1)
    max_features = np.max(drsprg_features, axis=1)
    velocity_features = np.diff(drsprg_features, axis=1).mean(axis=1)
    acceleration_features = np.diff(np.diff(drsprg_features, axis=1), axis=1).mean(axis=1)
    fft_features = np.abs(fft(drsprg_features, axis=1)).mean(axis=1)
    iqr_features = np.percentile(drsprg_features, 75, axis=1) - np.percentile(drsprg_features, 25, axis=1)


    # Stack into a single feature vector
    final_features = np.hstack([iqr_features, velocity_features, min_features, std_features, mean_features])

    df = pd.DataFrame(drsprg_flat)
    df["CKD_stage"] = drsprg_labels  # Add the CKD stage as the target column

    # Compute correlations
    correlations = df.corr()["CKD_stage"].drop("CKD_stage")  # Drop the target itself

    # Sort by absolute value to see the strongest correlations first
    sorted_correlation = correlations.abs().sort_values(ascending=False)

    # Print top features most correlated with CKD stage
    print("📊 Top Features Correlated with CKD Stage:")
    print(sorted_correlation.head(20))  # Change 20 to see more/less
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(drsprg_flat, drsprg_labels, test_size=test_size, random_state=42, shuffle=shuffle)
    
    # Print label counts
   
    #smote = SMOTE(sampling_strategy="auto", random_state=42)
    #X_train, y_train = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_models = 10
    selected_features = []
    models = []

    # Train multiple models with different random feature subsets
    for i in range(n_models):
        feature_subset = np.random.choice(X_train.shape[1], size=int(X_train.shape[1] * 0.8), replace=False)
        selected_features.append(feature_subset)

        model = SVC(kernel="linear")
        model.fit(X_train[:, feature_subset], y_train)
        models.append(model)

    # Make predictions (Ensemble Averaging)
    y_preds = np.array([model.predict(X_test[:, f]) for model, f in zip(models, selected_features)])
    y_final = np.round(y_preds.mean(axis=0))  # Majority Voting

    print("Ensemble Accuracy:", accuracy_score(y_test, y_final))
    svm_model = SVC(kernel="linear")

    # Forward Feature Selection
    sfs = SequentialFeatureSelector(estimator=svm_model, n_features_to_select=5, direction='forward')
    sfs.fit(X_train, y_train)
    selected_features = sfs.get_support(indices=True) # Selected feature indices

    print(selected_features)
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    # Train SVM
    svm_model.fit(X_train_selected, y_train)

    y_pred = svm_model.predict(X_test_selected)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 SVM Accuracy: {accuracy:.4f}")

    print(classification_report(y_test, y_pred))

    # Print classification report (Precision, Recall, F1-score)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 RF Accuracy: {accuracy:.4f}")

        # Create LightGBM datasets
    #lgb_train = lgb.Dataset(X_train, y_train)
    #lgb_test = lgb.Dataset(X_test, y_test)

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
    #lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test])

    # Predict
    #y_pred_lgb = lgb_model.predict(X_test)
    #y_pred_lgb_labels = np.argmax(y_pred_lgb, axis=1)  # Convert probabilities to class labels

    # Compute accuracy
    #lgb_acc = accuracy_score(y_test, y_pred_lgb_labels)
    #print("LightGBM Test Accuracy:", lgb_acc)

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

    #xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train the model
    #xgb_model.fit(X_train, y_train)

    # Make predictions
    #y_pred = xgb_model.predict(X_test)

    #accuracy = accuracy_score(y_test, y_pred)
    #print(f"📊 XGB Accuracy: {accuracy:.4f}")

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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def create_dataset_with_kfold(test_size: float = 0.2, random_state: int = 100, shuffle: bool = True, n_splits: int = 5):
    """Loads radiomics dataset, applies transformations & scaling BEFORE splitting, then trains an SVM model with Stratified K-Fold Cross-Validation."""
    
    # Load features and labels
    drsprg_features = np.load("features_drsprg_model.npy")
    drsprg_labels = np.load("labels_drsprg_model.npy")
    print(drsprg_features.shape)
    # Adjust labels (subtract 1 for 0-based indexing)
    drsprg_labels = drsprg_labels - 1

    # Flatten the features for training
    drsprg_flat = drsprg_features.reshape(drsprg_features.shape[0], -1)

    # Compute various feature transformations
    mean_features = np.mean(drsprg_features, axis=1)
    std_features = np.std(drsprg_features, axis=1)
    fft_features = np.abs(fft(drsprg_features, axis=1)).mean(axis=1)
    iqr_features = np.percentile(drsprg_features, 75, axis=1) - np.percentile(drsprg_features, 25, axis=1)

    #skew_features = skew(drsprg_features, axis=1)
    #kurtosis_features = kurtosis(drsprg_features, axis=1)
    final_features = np.hstack([fft_features, iqr_features, mean_features, std_features])

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Initialize variables for results
    accuracies = []
    svm_ensemble = []
    rf_accuracies = []
    logreg_accuracies = []
    bayes_accuracies = []
    knn_accuracies = []
    et_accuracies = []
    xgb_accuracies = []
    autogluon_accuracy = []

    # Perform Stratified K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(drsprg_flat, drsprg_labels)):
        print(f"Training fold {fold + 1}/{n_splits}...")
        

        # Split data into training and validation sets for this fold
        X_train, X_val = drsprg_flat[train_idx], drsprg_flat[val_idx]
        y_train, y_val = drsprg_labels[train_idx], drsprg_labels[val_idx]
        print(X_train.shape)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        #columns = [f"feature_{i}" for i in range(X_train.shape[1])]

        #X_train_pd = pd.DataFrame(X_train, columns=columns)  # Convert NumPy array to DataFrame
        #y_train_pd = pd.Series(y_train, name="CKD_stage")  # Convert NumPy array to Series

        #X_test_pd = pd.DataFrame(X_val, columns=columns)
        #y_test_pd = pd.Series(y_val, name="CKD_stage")

        # Reset indices to ensure alignment
        #X_train_pd = X_train_pd.reset_index(drop=True)
        #y_train_pd = y_train_pd.reset_index(drop=True)
        #X_test_pd = X_test_pd.reset_index(drop=True)
        #y_test_pd = y_test_pd.reset_index(drop=True)

        # Now concatenate safely
        #train_data = pd.concat([X_train_pd, y_train_pd], axis=1)
        #test_data = pd.concat([X_test_pd, y_test_pd], axis=1)

        # Verify alignment
        #print("Train Data After Fix:")
        #print(train_data.head())

        #predictor = TabularPredictor(label="CKD_stage", problem_type="multiclass").fit(train_data)

        # Evaluate AutoGluon
        #test_acc = predictor.evaluate(test_data)
        #print("AutoGluon Test Accuracy:", test_acc)
        #autogluon_accuracy.append(test_acc)

        # Make predictions (Ensemble Averaging)
        #y_preds = np.array([model.predict(X_val[:, f]) for model, f in zip(models, selected_features)])
        #y_final = np.round(y_preds.mean(axis=0))  # Majority Voting
        #accuracy = accuracy_score(y_val, y_final)
        #print("Ensemble Accuracy:", accuracy)
        #svm_ensemble.append(accuracy)

        # Train a model (e.g., SVM) on the current fold
        model = SVC(kernel="linear")
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate accuracy
        y_pred = model.predict(X_val)
        fold_accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(fold_accuracy)

        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        rf_accuracies.append(accuracy)
        print(f"📊 RF Accuracy: {accuracy:.4f}")

        logreg_model = LogisticRegression()
        logreg_model.fit(X_train, y_train)
        y_pred = logreg_model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        print(f"📊 LogReg Accuracy: {accuracy:.4f}")
        logreg_accuracies.append(accuracy)

        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        y_pred = nb_model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        bayes_accuracies.append(accuracy)
        print(f"📊 Bayes Accuracy: {accuracy:.4f}")

        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        knn_accuracies.append(accuracy)
        print(f"📊 KNN Accuracy: {accuracy:.4f}")

        et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        et_model.fit(X_train, y_train)
        y_pred = et_model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        et_accuracies.append(accuracy)
        print(f"📊 ExtraTreesClassifier Accuracy: {accuracy:.4f}")


        #xgb_model = XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=100)
        #xgb_model.fit(X_train, y_train)
        #y_pred = xgb_model.predict(X_val)

        #accuracy = accuracy_score(y_val, y_pred)
        #xgb_accuracies.append(accuracy)
        #print(f"📊 XGB CrossEntropy Accuracy: {accuracy:.4f}")

        
    # Print the overall cross-validation accuracy
    print(f"\nStratified K-Fold Cross-Validation Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"\nStratified K-Fold Cross-Validation Average Accuracy RF: {np.mean(rf_accuracies):.4f}")
    print(f"\nStratified K-Fold Cross-Validation Average Accuracy LogReg: {np.mean(logreg_accuracies):.4f}")
    print(f"\nStratified K-Fold Cross-Validation Average Accuracy Bayes: {np.mean(bayes_accuracies):.4f}")
    print(f"\nStratified K-Fold Cross-Validation Average Accuracy KNN: {np.mean(knn_accuracies):.4f}")
    print(f"\nStratified K-Fold Cross-Validation Average Accuracy ET: {np.mean(et_accuracies):.4f}")
    #print(f"\nStratified K-Fold Cross-Validation Average Accuracy SVM Ensemble: {np.mean(autogluon_accuracy):.4f}")
    #print(f"\nStratified K-Fold Cross-Validation Average Accuracy XGB: {np.mean(xgb_accuracies):.4f}")

# Run dataset creation with Stratified K-Fold cross-validation
create_dataset_with_kfold()

