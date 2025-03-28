import os
import pandas as pd
from pathlib import Path

"""
Sorts CSV files containing YOLO detection results or ground truth.
Sorting hierarchy:
1. Sequence (dataset)
2. Image ID
3. Bbox center coordinates (y_center, then x_center)

Input CSVs are read from data/label2csv and output to data/sortedcsv.
Handles the ground truth and detection results format.
"""

def sort_csv_files(input_dir: str, output_dir: str):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in input directory
    csv_files = list(Path(input_dir).glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    for csv_file in csv_files:
        try:
            # Read CSV file without headers
            df = pd.read_csv(csv_file, header=None)
            
            # Check number of columns to determine if it's ground truth or detection results
            num_columns = len(df.columns)
            
            # Ensure the ground truth has 7 columns and the detection results have 8 columns
            if num_columns == 7:
                df[7] = 1.0 # assign confidence score of 1.0 to ground truth
            elif num_columns != 8:
                raise ValueError(f"Unexpected number of columns ({num_columns}) in {csv_file.name}")
            
            # Name columns for easier handling
            df.columns = ['dataset', 'img_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence']
            
            # Convert img_id to integer for proper numerical sorting
            df['img_id'] = pd.to_numeric(df['img_id'])

            sorted_df = df.sort_values(
                by=['dataset', 'img_id', 'y_center', 'x_center'],
                ascending=[True, True, True, True]
            )
            
            # Save sorted CSV
            output_path = Path(output_dir) / csv_file.name
            sorted_df.to_csv(output_path, index=False, header=False)
            print(f"Saved sorted file to {output_path}")
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")

def main():
    # Define input and output directories
    input_dir = "data/label2csv"
    output_dir = "data/sortedcsv"
    
    sort_csv_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
