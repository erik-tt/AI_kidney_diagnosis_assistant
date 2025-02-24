import os
import re
import pandas as pd


# Use segmentation data for early testing as it includes
# masks that represent the segmentation and images. It is put together with
# labels from the csv files containing 

class FileReader:
    def __init__(self, base_dir:str, data_type:str):
        self.base_dir = base_dir
        self.data_type = data_type

    
    def get_pd_labels(self):
        #get csv data
        df_drsbru = pd.read_csv(os.path.join(self.base_dir, "labels/drsbru.csv"), delimiter=",")[['STUDY NAME', 'GENDER', 'CKD']]
        df_drsbrg = pd.read_csv(os.path.join(self.base_dir, "labels/drsprg.csv"), delimiter=",")[['STUDY NAME', 'GENDER', 'CKD']]
        df = pd.concat([df_drsbrg, df_drsbru])
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        df.dropna(inplace=True)

        return df


    def get_classification_data(self, folder_name : str =""):

        data = []
        #Change this when we have a good data structure
        path = os.path.join(self.base_dir, "dataset/drsprg", folder_name)

        df = self.get_pd_labels()

        walk_iter = os.walk(path)
        next(walk_iter)

        image_paths = []
        for root, _, files in walk_iter:
            absolute_path = os.path.abspath(root)

            for file in files:
                if ("image" in file) and (self.data_type == "image"):
                    image_paths.append(os.path.join(absolute_path, file))
                if (file.endswith(".dcm")) and (self.data_type == "time_series"):
                    image_paths.append(os.path.join(absolute_path, file))
            
        
        #Collaborated with chatGPT
        for image in image_paths:
            match = df[df['STUDY NAME'].apply(lambda study: study in image)]
            if not match.empty:
                label = match['CKD'].values[0]
                data.append({"image": image, "label": label})
    
        return data
