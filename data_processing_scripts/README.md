# Data Processing Scripts

Scripts used to take the results from the LISA x YOLO testing 

# PLEASE NOTE ALL SCRIPTS SHOULD BE RUN FROM PROJECT ROOT

### copy_proc_lisa.sh

Moves LISA files to correct location

### format_spark_verification.py
Summary of function:
The script processes Spark output files, specifically looking for part files (CSV files starting with part- and ending with .csv). It reads and combines these part files into a single DataFrame, then saves this combined data into a new CSV file. This is useful when Spark output is split into multiple part files and needs to be consolidated for further processing or analysis.

Input:
Part files: The individual part files from Spark, which are CSV files that need to be combined.

Output:
Combined CSV file (verification_counts_combined.csv): A single CSV file containing the concatenated data from all the part files

### light_counter.py
Summary of function:
This script uses Apache Spark to verify and count labels from various models (YOLOv3, YOLOv5, YOLOv8). It compares the counts of matched, missing, and misclassified labels for each model against a ground truth dataset. The script checks if the sum of matched and missing labels equals the total count from the ground truth

Input:
-Matched Data (CSV Files)
-Sorted CSV

Output:
It outputs a summary of these counts, along with a verification status (whether the counts match the ground truth), to both a text file and a CSV file.

### lisa_processed_label2csv.py
Summary of function:
This script converts YOLO model result text files into a CSV format. It processes YOLO detection results from several datasets, extracting label information from the YOLO output files, and writes this data into a structured CSV file (this is the "ground truth").

Input:
YOLO Results Path: The directory where YOLO output text files are stored for different datasets

Output:
lisa_processed_label.csv: A CSV file containing the processed labels from YOLO result text files. Each row in the CSV represents a detection from the YOLO model, with fields such as dataset name, file ID, and label information (like class and bounding box coordinates).


### lisa_yolo_val_label2csv.py
Summary of function:
Converts the labels of the validation/test data into a CSV format.

Input:
YOLO Results Path: The directory where YOLO output text files are stored for different datasets

Output:
A CSV file for each YOLO model, named <model_name>_output.csv. These files contain the processed label data from the YOLO text files, including dataset name, file ID, and detection details (class and bounding box coordinates).


### matching.py
Summary of function:
Compares the YOLO model predictions with ground truth data to classify detections into matched, misclassified, or missing categories based on the Intersection over Union (IoU) and Euclidean distance between bounding boxes.

Input:
CSV files with YOLO model predictions and ground truth labels

Output:
For each YOLO model:
-yolov{model}_matched.csv: CSV containing matched predictions from the model and ground truth based on IoU.
-yolov{model}_truth_matched.csv: CSV with ground truth boxes that were matched to model predictions.
-yolov{model}_misclassified.csv: CSV with model predictions that did not match any ground truth (misclassified).
-yolov{model}_missing.csv: CSV with ground truth boxes that did not have a matching model prediction (missing).


### sort_label_csvs.py
Summary of function:
Sorts all the CSVs rows by category, then img id, then bbox center coordinate.

Input:
CSV files containing either ground truth or detection results for YOLO

Output:
Sorted CSV files, the files are sorted based on dataset, image ID, and bounding box center coordinates.