import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#custom_string = "radiomics_features.npy"   # Example

# Process the 'imagepath' column
#def modify_path(original_path):
#    directory = os.path.dirname(original_path)
#    
    # Extract the folder name (second to last part)
#    folder_name = directory.split(os.sep)[-1]
#    
    # Build the new full path
#    new_path = os.path.join(directory, folder_name + "_radiomics.npz")
#    return new_path

# Create the new column
#metadata['RadiomicFeaturePath'] = metadata['ImagePath'].apply(modify_path)

# Save the updated metadata
#metadata.to_csv("../../../data/metadata.csv", index=False)


metadata = pd.read_csv("../../../data/metadata.csv")

# Initialize list to collect all feature arrays
all_features = []
all_labels = []

# Loop over each row
for idx, row in metadata.iterrows():
    try:
        npz_path = row['RadiomicFeaturePath']
        
        # Load the radiomic feature file
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract the feature values    
        features = data['feature_values']
        label = row['CKD']
        
        if pd.isna(label):
            print(f"Skipping row {idx} because CKD label is NaN")
            continue  # Skip this row entirely

        all_features.append(features)
        all_labels.append(label)
    except:
        print("failed")

# Convert to a big 2D NumPy array
X = np.vstack(all_features)  # Shape: (n_samples, n_features)
y = np.array(all_labels)     # Labels
print("Feature matrix shape:", X.shape)

mask_good_features = np.all(np.isfinite(X), axis=0)  # axis=0 for columns

# Apply the mask to X
X = X[:, mask_good_features]

print("Feature matrix shape:", X.shape)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# --- Get feature importances ---
importances = rf.feature_importances_

# Now we need feature names too
# Load feature names from the first npz file
first_data = np.load(metadata.loc[0, 'RadiomicFeaturePath'], allow_pickle=True)
feature_names = first_data['feature_names']
feature_names = feature_names[mask_good_features]

# Create a ranked DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df.to_csv("../../../data/feat_imp.csv", index=False)

# Print the top features
print(feature_importance_df)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for validation
    random_state=10,  # to make split reproducible
    stratify=y        # to preserve class balance (important for CKD data)
)

# --- Train Random Forest on TRAIN ONLY ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on validation set
y_pred = rf.predict(X_val)

# Calculate accuracy
from sklearn.metrics import accuracy_score
print("Validation Accuracy:", accuracy_score(y_val, y_pred))