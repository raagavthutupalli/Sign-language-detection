import os

def get_dataset_files(folder_path):
    if os.path.exists(folder_path):
        return os.listdir(folder_path)
    return []
