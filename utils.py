import os

def get_file_size(path):
    return round(os.path.getsize(path) / 1024, 2)
