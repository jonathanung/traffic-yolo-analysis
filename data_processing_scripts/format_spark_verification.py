import pandas as pd
import glob
import os
from pathlib import Path

def main():
    spark_output_dir = Path('data/counts/verification_counts.csv') 

    print(f"Looking for part files in: {spark_output_dir}")
    # Use glob to find all files starting with 'part-' and ending with '.csv' inside that directory
    part_files = glob.glob(str(spark_output_dir / 'part-*.csv'))

    if not part_files:
        print(f"ERROR: No part files found in the Spark output directory: {spark_output_dir}")
        print("Please ensure the light_counter.py script ran successfully and created output there.")
        return

    print(f"Found part files: {part_files}")

    # Read and concatenate all part files
    dfs = []
    try:
        for file in part_files:
            # Read the CSV, assuming it has headers as written by Spark
            df = pd.read_csv(file)
            dfs.append(df)
        
        if not dfs:
            print("ERROR: Failed to read any data from the part files.")
            return
            
        combined_df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        print(f"ERROR: Failed to read or concatenate part files. Error: {e}")
        return

    final_output_path = Path('data/counts') 
    final_output_path.mkdir(parents=True, exist_ok=True)

    output_file = final_output_path / 'verification_counts_combined.csv'
    try:
        combined_df.to_csv(output_file, index=False)
        print(f"Combined CSV saved successfully to: {output_file}")
    except Exception as e:
        print(f"ERROR: Failed to save combined CSV file. Error: {e}")


if __name__ == "__main__":
    main()
