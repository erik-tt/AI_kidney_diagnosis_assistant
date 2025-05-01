import os
import subprocess
import cv2
import nibabel as nib
import numpy as np
import shutil
import pandas as pd
import pydicom

input_dir = "../data/segmentation_masks"
output_dir = "../data/dataset"
dynamicrenal = "../data/BAZA dynamicrenal"
metadata = "../data/metadata.csv"

os.makedirs(output_dir, exist_ok=True)

    
def add_dicom_files(base_name, output_path):
    for root, dirs, files in os.walk(dynamicrenal):
                for filename in files:
                    if (base_name in filename) and filename.endswith(".dcm") and ("POST2" not in filename):
                        file_path = os.path.join(os.path.join(dynamicrenal, os.path.relpath(root, dynamicrenal)), filename)
                        if pydicom.dcmread(file_path).pixel_array.shape[0] > 100:
                            shutil.copy2(file_path, output_path)
                            time_series_path = os.path.join(output_path, filename)  # Save path to metadata
    return time_series_path

def convert_png_to_NIfTI(path, save_path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = np.expand_dims(image, axis=0)
    image_nii = nib.Nifti1Image(image, affine=np.eye(4))
    image_output_path = os.path.join(output_path, f"{base_name}_image.nii.gz")
    nib.save(image_nii, save_path)

def cleanup_excess_files(output_path):
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        try:
            if (os.path.isfile(item_path) or os.path.islink(item_path)) and not item_path.endswith(".nii.gz") and not item_path.endswith(".npz"):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error deleting {item_path}: {e}")

metadata_list = []

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".json"):
            relative_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(root, filename)
            output_path = os.path.join(output_folder, base_name)

            # Create img and label with LabelMe framework, saves image and label as png files
            subprocess.run(["labelme_export_json", json_path, "-o", output_path])
            
            image_path = os.path.join(output_path, "img.png")
            mask_path = os.path.join(output_path, "label.png")

            image_output_path = os.path.join(output_path, f"{base_name}_image.nii.gz")
            mask_output_path = os.path.join(output_path, f"{base_name}_label.nii.gz")

            # Convert the saved img and label from LabelMe as NIfTI files
            convert_png_to_NIfTI(image_path, image_output_path)
            convert_png_to_NIfTI(mask_path, mask_output_path)

            # Cleanup the generated files from LabelMe
            cleanup_excess_files(output_path)
        
            #Get dicom files. Long runtime, but this is only to be done once for setup.
            time_series_path = add_dicom_files(base_name, output_path)

            # Extract the patient ID and suffix (post, ant etc...)
            basename_split = base_name.split("_")
            database = basename_split[0]
            patient_id = "_".join(basename_split[:-1])
            suffix = basename_split[-1]

            # Save the data paths for easy lookup
            metadata_list.append({
                "ImageName": base_name,
                "PatientID": patient_id,
                "Suffix": suffix,
                "Database": database,
                "ImagePath": os.path.abspath(image_output_path),
                "SegLabelPath": os.path.abspath(mask_output_path),
                "TimeSeriesPath": os.path.abspath(time_series_path) if time_series_path else "Not Found",
            })

drsbru_labels = pd.read_csv("../data/labels/drsbru.csv")
drsprg_labels = pd.read_csv("../data/labels/drsprg.csv")

df_labels = pd.concat([drsbru_labels[["STUDY NAME", "CKD"]], drsprg_labels[["STUDY NAME", "CKD"]]], ignore_index=True)  # Resets index
df_labels["STUDY NAME"] = df_labels["STUDY NAME"].str.strip()

df_metadata = pd.DataFrame(metadata_list)
df_metadata = pd.merge(df_metadata, df_labels, how="left", left_on="PatientID", right_on="STUDY NAME")
df_metadata.drop(columns=["STUDY NAME"], inplace=True)
#Sort to get the dataset consistent across
df_metadata = df_metadata.sort_values(by="ImageName")

df_metadata.to_csv(metadata, index=False)