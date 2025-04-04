#F1 score but only using if IoU is over 45%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.ticker as mtick

# Define colors for consistency
MODEL_COLORS = {
    'YOLOv3': '#4472C4',
    'YOLOv5': '#ED7D31',
    'YOLOv8': '#70AD47'
}

def load_data(matched_csv_dir: Path) -> pd.DataFrame:
    """Loads matched, missing, and misclassified data for all models."""
    all_dfs = []
    models = ["3", "5", "8"]
    statuses = {'matched': 'Matched', 'missing': 'Missing', 'misclassified': 'Misclassified'}

    for model_num in models:
        model_version = f"YOLOv{model_num}"
        for status_key, status_name in statuses.items():
            file_path = matched_csv_dir / f"yolov{model_num}_{status_key}.csv"
            try:
                df = pd.read_csv(file_path)
                df['model_version'] = model_version
                df['status'] = status_name
                all_dfs.append(df[['model_version', 'dataset', 'status']])
            except FileNotFoundError:
                print(f"Warning: File not found - {file_path}. Skipping.")
            except pd.errors.EmptyDataError:
                print(f"Warning: File is empty - {file_path}. Skipping.")

    if not all_dfs:
        print("Error: No data loaded. Cannot calculate F1 scores.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)