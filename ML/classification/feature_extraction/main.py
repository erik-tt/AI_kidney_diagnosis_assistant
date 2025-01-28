from ML.segmentation.utils.file_reader import FileReader

file_reader = FileReader("../../../data")

drsprg_post = file_reader.get_image_data("drsprg/post", ["post", "label"])


