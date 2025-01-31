import os
import subprocess
import cv2
import nibabel as nib
import numpy as np
import shutil
import pydicom

input_dir = "../data/segmentation_masks"
output_dir = "../data/dataset"
dynamicrenal = "../data/BAZA dynamicrenal"

os.makedirs(output_dir, exist_ok=True)

def add_dicom_files(base_name, output_path):
    for root, dirs, files in os.walk(dynamicrenal):
                for filename in files:
                    if (base_name in filename) and filename.endswith(".dcm"):
                        file_path = os.path.join(os.path.join(dynamicrenal, os.path.relpath(root, dynamicrenal)), filename)
                        #if pydicom.dcmread(file_path).pixel_array.shape == (180, 128, 128):
                        shutil.copy2(file_path, output_path)


for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".json"):
            relative_path = os.path.relpath(root, input_dir)
            output_folder = os.path.join(output_dir, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(root, filename)
            output_path = os.path.join(output_folder, base_name)

            subprocess.run(["labelme_export_json", json_path, "-o", output_path])
            
            image_path = os.path.join(output_path, "img.png")
            mask_path = os.path.join(output_path, "label.png")

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            image_nii = nib.Nifti1Image(image, affine=np.eye(4))
            mask_nii = nib.Nifti1Image(mask, affine=np.eye(4))

            image_output_path = os.path.join(output_path, f"{base_name}_image.nii.gz")
            mask_output_path = os.path.join(output_path, f"{base_name}_label.nii.gz")

            for item in os.listdir(output_path):
                item_path = os.path.join(output_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error deleting {item_path}: {e}")

            nib.save(image_nii, image_output_path)
            nib.save(mask_nii, mask_output_path)

            #Get dicom files. Long runtime, but this is only to be done once for setup.
            add_dicom_files(base_name, output_path)