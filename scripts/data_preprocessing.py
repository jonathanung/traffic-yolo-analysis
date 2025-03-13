import os
import cv2
import numpy as np
import pandas as pd
import re
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


def read_annotations(annotation_path: str)->pd.DataFrame:
    """
    Given a path will return a pandas DataFrame with image Filename and bounding box coordinates
    """

    df = pd.read_csv(annotation_path, sep=';', usecols=['Filename', 'Upper left corner X', 'Upper left corner Y',
                                                        'Lower right corner X', 'Lower right corner Y'])

    df = df.rename(columns={
        'Upper left corner X': 'x_min',
        'Upper left corner Y': 'y_min',
        'Lower right corner X': 'x_max',
        'Lower right corner Y': 'y_max'
    })
    return df


def convert_to_yolo_format(df: pd.DataFrame, img_width: int, img_height: int) -> pd.DataFrame:
    """
    Convert bounding box coordinates to YOLO format.
    Returns a list of floats [x_center, y_center, width, height]
    """
    df['x_center'] = (df['x_min'] + df['x_max']) / (2 * img_width)
    df['y_center'] = (df['y_min'] + df['y_max']) / (2 * img_height)
    df['width'] = (df['x_max'] - df['x_min']) / img_width
    df['height'] = (df['y_max'] - df['y_min']) / img_height
    return df
    # x_min, y_min, x_max, y_max = bounding_box
    #
    # x_center = (x_min + x_max) / (2 * img_width)
    # y_center = (y_min + y_max) / (2 * img_height)
    # width = (x_max - x_min) / img_width
    # height = (y_max - y_min) / img_height
    #
    # return [x_center, y_center, width, height]


def preprocess_lisa_dataset(lisa_dir: str) -> None:
    """
    Preprocess LISA traffic light dataset:
    - Standardize image formats
    - Convert annotations to YOLO format
    - Split into train/val/test sets
    """
    # TODO: Implement LISA dataset preprocessing
    dataset_names = ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']

    for dataset in dataset_names:

        # call annotation reader function
        annotation_path = f"{lisa_dir}/Annotations/Annotations/{dataset}/frameAnnotationsBOX.csv"
        dataset_data = read_annotations(annotation_path)

        # convert to YOLO
        dataset_data = convert_to_yolo_format(dataset_data, 1280, 960)

        # strip img names of prefix directory
        dataset_data['Filename'] = dataset_data['Filename'].str.replace(fr'^.*?({dataset})', r'\1',
                                                                        regex=True)



        # TODO: split data into new YOLO train/validate sets

        print(dataset_data)

        # for dirpath, _, filenames in os.walk(dataset_dir):
        #     i = 0
        #     for IMGname in filenames:
        #         img_path = os.path.join(dirpath, IMGname)
        #         pattern = fr'{IMGname}'
        #         if dataset_data['Filename'].str.contains(pattern, na=False, regex=True).any() and i < 5:
        #             print(img_path)
        #         i += 1

    # img_width, img_height = get_image_size(###)
    #
    # convert_to_yolo_format(img_width,img_height,)
    # pass


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
