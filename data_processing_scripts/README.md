# Data Processing Scripts

Scripts used to take the results from the LISA x YOLO testing 

# PLEASE NOTE ALL SCRIPTS SHOULD BE RUN FROM PROJECT ROOT

### lisa_yolo_val_label2csv.py

Converts the labels of the validation/test data into a CSV format

### lisa_processed_label2csv.py

Converts the labels of the processed data into a CSV format (this is the "ground truth")

### sort_label_csvs.py

Sorts all the CSVs rows by category, then img id, then bbox center coordinate.