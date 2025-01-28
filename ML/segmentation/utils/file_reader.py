import os
import glob
import re

class FileReader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    # Input: relative_path, files from relative path, return dictionary

    def get_image_data(self, folder_name, file_suffixes):
        path = os.path.join(self.base_dir, "dataset", folder_name)

        
        patterns = [
            re.compile(rf"^[a-zA-Z]+_\d+_{re.escape(s)}(?:_[a-zA-Z]+)?\.(dcm|nii\.gz)$", re.IGNORECASE)
            for s in file_suffixes
        ]

        print(patterns)
                                       
        for root, _, files in os.walk(path):
            #s = ' '.join(files) #mulig raskere?
            for file in files:
                for pattern in patterns:
                    #print(file)
                    if pattern.match(file):  # Use `match` for filename-level patterns
                        print(f"Matched: {file}")
            
            #re.findall(s, patterns)

                #print(file)
        #pattern = re.compile(r"^[a-zA-Z]+_\d+_" + re.escape(file_suffix) + r"\.dcm|nii\.gz$", re.IGNORECASE)
        
    def get_data_file_paths(self, folder_name, suffix):
        file_paths = []
        pattern = re.compile(r"^[a-zA-Z]+_\d+_" + re.escape(suffix) + r"\.dcm$", re.IGNORECASE)

        full_path = os.path.normpath(os.path.join(self.base_dir,"BAZA dynamicrenal", folder_name,"DATA_DICOM"))
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.endswith(".dcm") and pattern.match(file):
                    full_file_path = os.path.join(root, file)
                    file_paths.append(full_file_path)

        return file_paths
    
    def get_segmentation_file_paths(self, folder_name):
        data = []
        path = os.path.join(self.base_dir, "dataset", folder_name)

        walk_iter = os.walk(path)
        next(walk_iter)

        for root, _, files in walk_iter:
            absolute_path = os.path.abspath(root)
            
            image_path = None
            label_path = None

            for file in files:
                print(absolute_path)
                print(file)
                if "image" in file:
                    image_path = os.path.join(absolute_path, file)
                elif "label" in file:
                    label_path = os.path.join(absolute_path, file)
                elif "dcm" in file:
                    continue
                else:
                    continue
                
        
            data.append({"image": image_path, "label": label_path})
        
        return data

