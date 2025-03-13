import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image

def get_image_size(image_path: str) -> tuple:
    """
    Retrieve the actual width and height of an image using Pillow.
    """
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)

def read_annotations(annotation_path: str):
    """
    Given a path will return a pandas DataFrame with image Filename and bounding box coordinates
    """
    annotation_path += "frameAnnotationsBOX.csv"

    return pd.read_csv(annotation_path, sep=';', usecols=['Filename', 'Upper left corner X', 'Upper left corner Y',
                                                   'Lower right corner X', 'Lower right corner Y'])

def convert_to_yolo_format(img_width: int, img_height: int, bounding_box: List[int]) -> List[float]:
    """
    Convert bounding box coordinates to YOLO format.
    Returns a list of floats [x_center, y_center, width, height]
    """
    
    x_min, y_min, x_max, y_max = bounding_box
    
    x_center = (x_min + x_max) / (2 * img_width)
    y_center = (y_min + y_max) / (2 * img_height)
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [x_center, y_center, width, height]

def preprocess_lisa_dataset(lisa_dir: str) -> None:
    """
    Preprocess LISA traffic light dataset:
    - Standardize image formats
    - Convert annotations to YOLO format
    - Split into train/val/test sets
    """
    # TODO: Implement LISA dataset preprocessing
    img_width, img_height = get_image_size(###)
    
    convert_to_yolo_format(img_width,img_height,)
    pass

def preprocess_bosch_dataset(bosch_dir: str) -> None:
    """
    Preprocess Bosch Small Traffic Lights dataset:
    - Standardize image formats 
    - Convert annotations to YOLO format
    - Split into train/val/test sets
    """
    # TODO: Implement Bosch dataset preprocessing
    pass

def main():
    os.makedirs("data/processed", exist_ok=True)
    
    lisa_dir: str = "data/raw/lisa"
    bosch_dir: str = "data/raw/bosch"
    
    if os.path.exists(lisa_dir):
        preprocess_lisa_dataset(lisa_dir)
    
    if os.path.exists(bosch_dir):
        preprocess_bosch_dataset(bosch_dir)

if __name__ == "__main__":
    main() 