import os
import subprocess
import cv2
import nibabel as nib
import numpy as np
import shutil

input_dir = "../../../data/segmentation_masks"
output_dir = "../../../data/segmentation_dataset"

os.makedirs(output_dir, exist_ok=True)

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