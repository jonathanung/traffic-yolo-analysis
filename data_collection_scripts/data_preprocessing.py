import os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image
import shutil
import yaml


def create_yaml(output_dir: str, dataset_name: str) -> None:
    # Create the dataset directory if it doesn't exist
    dataset_path = f"{output_dir}/{dataset_name}"
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    img_path = f"{output_dir}/{dataset_name}/images"
    labels_path = f"{output_dir}/{dataset_name}/labels"

    # Get class names (or use a default list) 
    class_names = ["traffic_light"]

    yaml_data = {
        "train_img": img_path,
        "train_label": labels_path,
        "nc": len(class_names),
        "names": class_names,
    }

    dataset_path = f"{output_dir}/{dataset_name}"
    yaml_path = Path(dataset_path) / f"{dataset_name}.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

    print(f"Generated: {yaml_path}")




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

        # sort df by fname
        dataset_frame = dataset_frame.sort_values(by='Filename')

        print(dataset_frame)

        # FOR DEBUGGING: export df to csv
        # dataset_frame.to_csv('dataset_frame.csv', index=False)

        ### COULD BE DELETED ###

        export_yolo_data(lisa_dir, 'data/processed/lisa/yolo', dataset_name, dataset_frame)


def export_yolo_data(lisa_dir: str, output_dir: str, dataset_name: str, df: pd.DataFrame) -> None:
    """
    Export YOLO data to a directory by copying images directly from the raw folder.
    - Copy images to output_dir/{dataset_name}/images/...
    - For dayTrain and nightTrain, place all subdirectory images under the same parent
    - Create directory structure for YOLO training
    """
    create_yaml(output_dir, dataset_name)
    # Create images and label dataset-specific directories
    os.makedirs(f"{output_dir}/{dataset_name}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/{dataset_name}/labels", exist_ok=True)
    
    # Find all images in the dataset directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    # Special handling for dayTrain and nightTrain
    if dataset_name in ['dayTrain', 'nightTrain']:
        # For training datasets, look for all clips under the training directory
        train_dir = Path(f"{lisa_dir}/{dataset_name}/{dataset_name}")
        if train_dir.exists():
            # Find all clip directories
            clip_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
            
            # Find images in each clip directory
            for clip_dir in clip_dirs:
                frames_dir = clip_dir / "frames"
                if frames_dir.exists():
                    for ext in image_extensions:
                        image_files.extend(list(frames_dir.glob(f"*{ext}")))
    else:
        # For sequence datasets, use the standard approach
        for ext in image_extensions:
            possible_paths = [
                Path(f"{lisa_dir}/{dataset_name}/{dataset_name}/frames"),
                Path(f"{lisa_dir}/{dataset_name}/frames"),
                Path(f"{lisa_dir}/{dataset_name}")
            ]
            
            for path in possible_paths:
                if path.exists():
                    image_files.extend(list(path.glob(f"*{ext}")))
    
    print(f"Found {len(image_files)} images for {dataset_name}")
    
    # Copy images to output directory
    copied_count = 0
    for img_path in image_files:
        # Get just the filename without path
        filename = img_path.name

        # Destination path - all images go directly under the dataset directory
        dst_path = f"{output_dir}/{dataset_name}/images/{filename}"

        # Copy the image
        try:
            shutil.copy2(img_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {img_path} to {dst_path}: {e}")

    print(f"Exported {copied_count} images to {output_dir}/images/{dataset_name}")


    # Group by filename to handle multiple objects in the same image
    grouped = df.groupby('Filename')
    label_count = 0
    labels_dir = f"{output_dir}/{dataset_name}/labels"

    for filename, group in grouped:

        # Remove file extension if present
        base_filename = filename.split('.')[0] if '.' in filename else filename


        # Create label file path
        label_path = f"{labels_dir}/{base_filename}.txt"
        # Write all objects for this image to the label file
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                # Format: class x_center y_center width height
                f.write(f"{int(row['object_class'])} {row['x_center']} {row['y_center']} {row['width']} {row['height']}\n")
                label_count += 1
    
    print(f"Exported {label_count} object labels across {len(grouped)} files to {labels_dir}")

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
