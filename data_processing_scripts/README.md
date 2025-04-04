# Data Processing Scripts

Scripts used to take the results from the LISA x YOLO testing 

# PLEASE NOTE ALL SCRIPTS SHOULD BE RUN FROM PROJECT ROOT

### copy_proc_lisa.sh

Moves LISA files to correct location

### format_spark_verification.py
Summary of function:

Input:

Output:


### light_counter.py
Summary of function:

Input:

Output:


### lisa_processed_label2csv.py
Summary of function:
Converts the labels of the processed data into a CSV format (this is the "ground truth")

Input:

Output:


### lisa_yolo_val_label2csv.py
Summary of function:
Converts the labels of the validation/test data into a CSV format

Input:

Output:


### matching.py
Summary of function:
Matches rows in the CSVs from model to truth, and outputs CSVs for data analysis.

Input:

Output:


### sort_label_csvs.py
Summary of function:
Sorts all the CSVs rows by category, then img id, then bbox center coordinate.

Input:

Output:
