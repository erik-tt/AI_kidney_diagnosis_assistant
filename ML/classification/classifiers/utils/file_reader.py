import os
import re
import pandas as pd


# Use segmentation data for early testing as it includes
# masks that represent the segmentation and images. It is put together with
# labels from the csv files containing 

class FileReader:
    def __init__(self, base_dir:str):
        self.base_dir = base_dir

    
    def get_pd_labels(self):
        #get csv data
        df_drsbru = pd.read_csv(os.path.join(self.base_dir, "classification_labels/drsbru.csv"), delimiter=",")[['STUDY NAME', 'GENDER', 'CKD']]
        df_drsbrg = pd.read_csv(os.path.join(self.base_dir, "classification_labels/drsprg.csv"), delimiter=",")[['STUDY NAME', 'GENDER', 'CKD']]
        df = pd.concat([df_drsbrg, df_drsbru])
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        df.dropna(inplace=True)

        return df


    def get_classification_data(self, folder_name : str =""):

        data = []
        #Change this when we have a good data structure
        path = os.path.join(self.base_dir, "segmentation_dataset/", folder_name)

        df = self.get_pd_labels()

        walk_iter = os.walk(path)
        next(walk_iter)

        image_paths = []
        for root, _, files in walk_iter:
            absolute_path = os.path.abspath(root)

            for file in files:
                if "image" in file:
                    image_paths.append(os.path.join(absolute_path, file))
            
        
        #Collaborated with chatGPT
        for image in image_paths:
            match = df[df['STUDY NAME'].apply(lambda study: study in image)]
            if not match.empty:
                label = match['CKD'].values[0]
                data.append({"image": image, "label": label})
    
        return data
