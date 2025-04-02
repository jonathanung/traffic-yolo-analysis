#! /bin/bash

# These script takes the resulting data and processes it into relevant CSV files for the data analysis

# Convert the YOLO txt files to CSV files
python3 ./data_processing_scripts/lisa_yolo_val_label2csv.py

# Convert the Processed YOLOxLISA data to a CSV file
python3 ./data_processing_scripts/lisa_processed_label2csv.py

# The two scripts above are to be run first, because the matching script relies on the output of the two above

# Sort the label CSVs for processing
python3 ./data_processing_scripts/sort_label_csvs.py

# Match the YOLO and Processed YOLOxLISA data
python3 ./data_processing_scripts/matching.py

# Copy the matched CSV files to the data/matched_csv directory
bash ./data_processing_scripts/copy_proc_lisa.sh

# Count the number of lights in the data
python3 ./data_processing_scripts/light_counter.py

# Format the Spark output
python3 ./data_processing_scripts/format_spark_verification.py