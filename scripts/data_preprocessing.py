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


def read_data(annotation_path: str, df: pd.DataFrame, img_path: str)->pd.DataFrame:
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
    img_width, img_height = get_image_size(img_path)
    new_df['img_width'] = img_width
    new_df['img_height'] = img_height

    # Extract filename from path
    new_df['filetype'] = new_df['Filename'].apply(lambda x: x.split('.')[-1])

    new_df['Filename'] = new_df['Filename'].apply(lambda x: x.split('/')[-1] if '/' in x else x).apply(lambda x: x.split('.')[0])

    # from img_path, remove everything after trailing '/'
    img_path = img_path.split('/')[:-1]
    img_path = '/'.join(img_path) + '/'
    new_df['img_path'] = img_path


    if not df.empty:
        df = pd.concat([df, new_df])
    else:
        df = new_df
    return df


def convert_to_yolo_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert bounding box coordinates to YOLO format.
    Returns a list of floats [x_center, y_center, width, height]
    """
    df['x_center'] = (df['x_min'] + df['x_max']) / (2 * df['img_width'])
    df['y_center'] = (df['y_min'] + df['y_max']) / (2 * df['img_height'])
    df['width'] = (df['x_max'] - df['x_min']) / df['img_width']
    df['height'] = (df['y_max'] - df['y_min']) / df['img_height']
    # add object class column with zeros to determine classification type (and since its only traffic lights, 0)
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
            img_path = f"{lisa_dir}/{dataset_name}/{dataset_name}/frames/{dataset_name}--00000.jpg"
            dataset_frame = read_data(annotation_path, dataset_frame, img_path)

        else:
            # walk through all subdirectories in the train directory
            for _, dirs, _ in os.walk(f"{lisa_dir}/Annotations/Annotations/{dataset_name}"):
                for dir in dirs:
                    if dir not in walked_dirs:
                        walked_dirs.append(dir)
                        annotation_path = f"{lisa_dir}/Annotations/Annotations/{dataset_name}/{dir}/frameAnnotationsBOX.csv"
                        img_path = f"{lisa_dir}/{dataset_name}/{dataset_name}/{dir}/frames/{dir}--00000.jpg"
                        dataset_frame = read_data(annotation_path, dataset_frame, img_path)


        # convert to YOLO
        dataset_frame = convert_to_yolo_format(dataset_frame)

        img_file_names = dataset_frame['Filename'].values

        print(dataset_frame)

        # FOR DEBUGGING: export df to csv
        dataset_frame.to_csv('dataset_frame.csv', index=False)

        # TODO: split data into new YOLO train/validate sets


        ### COULD BE DELETED ###

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
