# Scripts

This directory contains various utility scripts for the traffic light detection project.

## Data Processing Scripts

### download.py

Downloads the LISA and Bosch traffic light datasets from Kaggle. 

Follow instructions in the [DATA.md](../DATA.md) file to set up the Kaggle API.

### data_preprocessing.py

Preprocesses the LISA traffic light dataset to be used for training YOLO models.

For future: Process BOSCH dataset

## Training Scripts

### train_lisa.py

Trains YOLO models on the LISA traffic light dataset.

### Standalone Test Scripts

We provide individual test scripts for each YOLO version to ensure reliable testing:

#### yolo3_test.py

Standalone script for testing YOLOv3 models. Features:
- Confidence threshold: 0.25
- IoU threshold: 0.45
- Supports visualization and text output
- Handles both trained and pretrained models

#### yolo5_test.py

Standalone script for testing YOLOv5 models. Features:
- Confidence threshold: 0.25
- IoU threshold: 0.45
- Compatible with latest YOLOv5 versions
- Supports multiple test sequences

#### yolo8_test.py

Standalone script for testing YOLOv8 models. Features:
- Confidence threshold: 0.25
- IoU threshold: 0.45
- Uses Ultralytics YOLO package
- Simplified testing workflow
