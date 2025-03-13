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


def read_annotations(annotation_path: str, df: pd.DataFrame)->pd.DataFrame:
    """
    Given a path will return a pandas DataFrame with image Filename and bounding box coordinates
    """
    new_df = pd.read_csv(annotation_path, sep=';', usecols=['Filename', 'Upper left corner X', 'Upper left corner Y',
                                                    'Lower right corner X', 'Lower right corner Y'])
    new_df = new_df.rename(columns={
        'Upper left corner X': 'x_min',
        'Upper left corner Y': 'y_min',
        'Lower right corner X': 'x_max',
        'Lower right corner Y': 'y_max'
    })
    if not df.empty:
        df = pd.concat([df, new_df])
    else:
        df = new_df
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
    df['object_class'] = 0
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

    datasets = [
        {'name' : 'daySequence1', 'train' : False},
        {'name' : 'daySequence2', 'train' : False},
        {'name' : 'dayTrain', 'train' : True},
        {'name' : 'nightSequence1', 'train' : False},
        {'name' : 'nightSequence2', 'train' : False},
        {'name' : 'nightTrain', 'train' : True},
    ]
    
    # dataset_names = ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']

    for dataset_struct in datasets:
        dataset_name: str = dataset_struct['name']
        train: bool = dataset_struct['train']
        walked_dirs : List[str] = []

        dataset_frame = pd.DataFrame()
        # call annotation reader function
        if not train:
            annotation_path = f"{lisa_dir}/Annotations/Annotations/{dataset_name}/frameAnnotationsBOX.csv"
            dataset_frame = read_annotations(annotation_path, dataset_frame)
        else:
            # walk through all subdirectories in the train directory
            for root, dirs, files in os.walk(f"{lisa_dir}/Annotations/Annotations/{dataset_name}"):
                for dir in dirs:
                    if dir not in walked_dirs:
                        walked_dirs.append(dir)
                        annotation_path = f"{lisa_dir}/Annotations/Annotations/{dataset_name}/{dir}/frameAnnotationsBOX.csv"
                        dataset_frame = read_annotations(annotation_path, dataset_frame)

        # convert to YOLO
        dataset_frame = convert_to_yolo_format(dataset_frame, 1280, 960)

        # strip img names of prefix directory
        # if not train:
        #     dataset_frame['Filename'] = dataset_frame['Filename'].str.replace(fr'^.*?({dataset_name})', r'\1',
        #                                                                 regex=True)
        # else:
        #     dataset_frame['Filename'] = dataset_frame['Filename'].str.replace(fr'^.*?({dataset_name})', r'\1',
        #                                                                 regex=True)

        img_file_names = dataset_frame['Filename'].values

        print(dataset_frame)

        # TODO: split data into new YOLO train/validate sets


        ### COULD BE DELETED ###
        # inserting class column with zereos
        dataset_frame['object_class'] = 0

        # prep df to be exported as .txt
        dataset_frame = dataset_frame[['object_class', 'x_center', 'y_center', 'width', 'height']]


        # TODO: discuss where to put annotation .txt files and where to put YOLO datasets
        ### EXPORTING INTO CURRENT DIRECTORY ###
        # for img in img_file_names:
        #     img = img[:-4]
        #     img += '.txt'
        #     dataset_data.to_csv(img,sep=' ',index=False, header=False)
        #     break

        # print(dataset_data)



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
