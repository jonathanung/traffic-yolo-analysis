#! /bin/bash

# These script takes the resulting data and processes it into relevant CSV files for the data analysis

# Convert the YOLO txt files to CSV files
python ./data_processing_scripts/lisa_yolo_val_label2csv.py

# Convert the Processed YOLOxLISA data to a CSV file
python ./data_processing_scripts/lisa_processed_label2csv.py

# The two scripts above are to be run first, because the matching script relies on the output of the two above

# Sort the label CSVs for processing
python ./data_processing_scripts/sort_label_csv.py

# Match the YOLO and Processed YOLOxLISA data
python ./data_processing_scripts/matching.py
