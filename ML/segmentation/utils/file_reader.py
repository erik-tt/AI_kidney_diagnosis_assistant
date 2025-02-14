import os
import re

class FileReader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

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
                if "image" in file:
                    image_path = os.path.join(absolute_path, file)
                elif "label" in file:
                    label_path = os.path.join(absolute_path, file)
                elif "dcm" in file:
                    continue
                else:
                    raise ValueError("data does not contain label or image")
                
        
            data.append({"image": image_path, "label": label_path})
        
        return data

