import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.ticker as mtick

'''
Should aim to find the residuals between the matches and the ground truth.
    1. Load in models results into a dataframe
    2. Find the residuals between the matches and the ground truth for each model
    3. Plot the residuals for each model
'''

def load_data(matched_csv_dir: Path, models:list, file_content:str)->pd.DataFrame:
    '''
    given a path should read in all the matched csv files and concat all the models into a single dataframe.
    '''
    all_dfs = []
    for model in models:
        model_version = f"YoloV{model}"
        file = model_version.lower()
        file += file_content
        file_path = matched_csv_dir / file

        try:
            df = pd.read_csv(file_path)
            df.insert(0, 'Model', model_version)
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"Warning: File is empty - {file_path}. Skipping.")

    if not all_dfs:
        print("Error: No data loaded. Cannot generate plots.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def calculate_residuals(results: pd.DataFrame, ground_truth: pd.DataFrame)->pd.DataFrame:
    residual= results[["Model", "dataset", "img_id"]].copy()
    residual["Residual_x_center"] = results["x_center"] - ground_truth["x_center"]
    residual["Residual_y_center"] = results["y_center"] - ground_truth["y_center"]
    residual["Residual_width"] = results["width"] - ground_truth["width"]
    residual["Residual_height"] = results["height"] - ground_truth["height"]

    return residual


#TODO: make function to plot out df data


def main():
    matched_csv_dir = Path("./data/matched_csv")
    models = ["3", "5", "8"]

    results_data = load_data(matched_csv_dir, models, "_matched.csv")
    truth_data = load_data(matched_csv_dir, models, "_truth_matched.csv")

    residuals_data = calculate_residuals(results=results_data, ground_truth=truth_data)

    print(residuals_data)



if __name__ == "__main__":
    main()