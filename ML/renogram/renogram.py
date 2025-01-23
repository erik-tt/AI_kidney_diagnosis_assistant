import nibabel as nib
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# For testing
#img = "../../data/BAZA dynamicrenal/drsbru/DATA_DICOM/drsbru_004/drsbru_004_POST.dcm"
#mask = "../../data/segmentation_dataset/drsbru/post/drsbru_004_POST/drsbru_004_POST_label.nii.gz"

def generate_renogram(image_series, mask):
    nii_file = nib.load(mask)

    label_data = nii_file.get_fdata()
    label_data = np.squeeze(label_data)

    dicom_data = pydicom.dcmread(image_series)
    pixel_array = dicom_data.pixel_array

    roi_labels = np.unique(label_data)
    roi_labels = roi_labels[roi_labels != 0]  

    roi_averages = {label: [] for label in roi_labels}

    for t in range(pixel_array.shape[0]):
        img = pixel_array[t, :, :]
        
        for label in roi_labels:
            roi_mask = (label_data == label)
            roi_pixels = img[roi_mask]

            if roi_pixels.size > 0:
                avg_intensity = np.mean(roi_pixels)
            else:
                avg_intensity = 0
            
            roi_averages[label].append(avg_intensity)

    # Kan plotte, men burde slettes og kun returne roi_averages
    for label, intensities in roi_averages.items():
        plt.plot(intensities, label=f"ROI {label}")

    # Oversette fra time frames til minutter
    plt.xlabel("Time Frame")
    plt.ylabel("Average Intensity")
    plt.title("Renogram (Time-Activity Curve)")
    plt.legend()
    plt.show()

# For testing
#generate_renogram(img, mask)