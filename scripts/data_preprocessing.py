import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List, Dict

def convert_to_yolo_format(img_width: int, img_height: int, bounding_box: List[int]) -> List[float]:
    """
    Convert bounding box coordinates to YOLO format.
    Returns a list of floats [x_center, y_center, width, height]
    """
    pass

def preprocess_lisa_dataset(lisa_dir: str) -> None:
    """
    Preprocess LISA traffic light dataset:
    - Standardize image formats
    - Convert annotations to YOLO format
    - Split into train/val/test sets
    """
    # TODO: Implement LISA dataset preprocessing
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