from radiomics.featureextractor import RadiomicsFeatureExtractor
import numpy as np
import SimpleITK as sitk

img = "../../../data/BAZA dynamicrenal/drsbru/DATA_DICOM/drsbru_004/drsbru_004_POST.dcm"
mask = "../../../data/segmentation_dataset/drsbru/post/drsbru_004_POST/drsbru_004_POST_label.nii.gz"

def extract_radiomic_features(image_series, mask):
    image = sitk.ReadImage(image_series)  
    mask = sitk.ReadImage(mask)  

    image_array = sitk.GetArrayFromImage(image)  
    mask_array = sitk.GetArrayFromImage(mask) 
    mask_array = mask_array.squeeze()

    extractor = RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape2D')
    
    radiomic_features = {}
    for region in np.unique(mask_array):
        if region == 0:
            continue
        
        radiomic_features[region] = {}

        extractor.settings['label'] = region

        for t in range(image_array.shape[0]):
            image_slice = sitk.GetImageFromArray(image_array[t])
            mask_slice = sitk.GetImageFromArray(mask_array)

            results = extractor.execute(image_slice, mask_slice)
            radiomic_features[region][t] = results

    return radiomic_features

extract_radiomic_features(img, mask)