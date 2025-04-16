import os
import yaml
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_DIRECTORIES = [
    "outputs",
    "uploads",
    "weights",
    "yolov5/runs"
]

def is_path_allowed(file_path):
    normalized_path = os.path.normpath(file_path)
    
    if os.path.isabs(normalized_path):
        try:
            relative_path = os.path.relpath(normalized_path, BASE_DIR)
            if relative_path.startswith('..'):
                return False
        except ValueError:
            return False
    else:
        relative_path = normalized_path
    
    for allowed_dir in ALLOWED_DIRECTORIES:
        allowed_dir_norm = os.path.normpath(allowed_dir)
        
        if (relative_path == allowed_dir_norm or 
            relative_path.startswith(allowed_dir_norm + os.sep)):
            return True
    
    return False

def get_absolute_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(BASE_DIR, relative_path)

def create_yaml_config(dataset_path, num_classes, class_names, train_path, val_path):
    """
    Create a dataset configuration YAML file for YOLOv5 training
    """
    config = {
        'path': os.path.abspath(dataset_path),
        'train': train_path,
        'val': val_path,
        'nc': num_classes,
        'names': class_names
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(dataset_path, f'{timestamp}_data_config.yaml')
    with open(config_path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    return config_path
