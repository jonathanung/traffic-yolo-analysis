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
    # create a list that will store each models df
    all_dfs = []

    for model in models:
        model_version = f"YoloV{model}"
        file = model_version.lower()
        file += file_content
        file_path = matched_csv_dir / file

        try:
            # read in csv to df
            df = pd.read_csv(file_path)
            # insert the model number which will be used to separate the data later in plotting
            df.insert(0, 'Model', model_version)
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"Warning: File is empty - {file_path}. Skipping.")

    # if DF came back empty
    if not all_dfs:
        print("Error: No data loaded. Cannot generate plots.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def calculate_IoU(results: pd.DataFrame, ground_truth: pd.DataFrame)->pd.DataFrame:
    IoU = pd.DataFrame()

    # Fail conditions when either df is empty or when their lengths are disproportional
    if results.empty or ground_truth.empty:
        print(f"ERROR: either results data or ground truth data is missing. "
              f"Results missing:{results.empty}. Truth data:{ground_truth.empty} ")
        return IoU

    if len(results) != len(ground_truth):
        print("ERROR: Number of results matched and ground truth matched do not line up. Please run check process_data.sh")
        return IoU

    # copy columns from df
    IoU= results[["Model", "dataset", "img_id", "confidence"]].copy()


    # box area calculations
    IoU["predicted_box_area"] = results["height"] * results["width"]
    IoU["true_box_area"] = ground_truth["height"] * ground_truth["width"]

    # predicted x_min. x_max, y_min, y_max
    IoU["predicted_x_min"] = results['x_center'] - results['width']
    IoU["predicted_x_max"] = results['x_center'] + results['width']
    IoU["predicted_y_min"] = results['y_center'] - results['height']
    IoU["predicted_y_max"] = results['y_center'] + results['height']

    # true x_min. x_max, y_min, y_max
    IoU["true_x_min"] = ground_truth['x_center'] - ground_truth['width']
    IoU["true_x_max"] = ground_truth['x_center'] + ground_truth['width']
    IoU["true_y_min"] = ground_truth['y_center'] - ground_truth['height']
    IoU["true_y_max"] = ground_truth['y_center'] + ground_truth['height']

    # calculate intersections on corners of bounding mox
    IoU["inter_x_min"] = IoU[["predicted_x_min", "true_x_min"]].max(axis=1)
    IoU["inter_x_max"] = IoU[["predicted_x_max", "true_x_max"]].max(axis=1)
    IoU["inter_y_min"] = IoU[["predicted_y_min", "true_y_min"]].max(axis=1)
    IoU["inter_y_max"] = IoU[["predicted_y_max", "true_y_max"]].max(axis=1)

    # Compute intersections width and height
    IoU["inter_width"] = (IoU["inter_x_max"] - IoU["inter_x_min"]).clip(lower=0)
    IoU["inter_height"] = (IoU["inter_y_max"] - IoU["inter_y_min"]).clip(lower=0)

    # compute the overlapping area
    IoU["intersecting area"] = IoU["inter_width"] * IoU["inter_height"]

    return IoU[["Model", "dataset", "img_id", "intersecting area", "confidence"]]




def calculate_euclidean_distances(results: pd.DataFrame, ground_truth: pd.DataFrame)-> pd.DataFrame:
    '''
    returns euclidean distance
    '''
    euclidian_data = pd.DataFrame()

    # Fail conditions when either df is empty or when their lengths are disproportional
    if results.empty or ground_truth.empty:
        print(f"ERROR: either results data or ground truth data is missing. "
              f"Results missing:{results.empty}. Truth data:{ground_truth.empty} ")
        return euclidian_data

    if len(results) != len(ground_truth):
        print("ERROR: Number of results matched and ground truth matched do not line up. Please run check process_data.sh")
        return euclidian_data

    # copy columns from df
    euclidian_data= results[["Model", "dataset", "img_id", "confidence"]].copy()
    euclidian_data["euc_distance"] = np.sqrt((ground_truth["x_center"] - results["x_center"]) ** 2 +
                                             (ground_truth["y_center"] - results["y_center"]) ** 2
                                             )

    return euclidian_data[["Model", "dataset", "img_id", "euc_distance", "confidence"]]



#TODO: make function to plot out df data
def plot_data(data:pd.DataFrame):






def main():
    matched_csv_dir = Path("./data/matched_csv")
    models = ["3", "5", "8"]

    results_data = load_data(matched_csv_dir, models, "_matched.csv")
    truth_data = load_data(matched_csv_dir, models, "_truth_matched.csv")

    IoU_data = calculate_IoU(results=results_data, ground_truth=truth_data)
    euclidian_data = calculate_euclidean_distances(results=results_data, ground_truth=truth_data)

    print(IoU_data)
    print(euclidian_data)



if __name__ == "__main__":
    main()