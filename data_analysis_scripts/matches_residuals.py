import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.ticker as mtick
import seaborn as sns
from pandas.core.interchange.dataframe_protocol import DataFrame

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


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in YOLO format (x_center, y_center, width, height)
    """
    # Convert from center format to corner format
    box1_x1 = box1['x_center'] - box1['width'] / 2
    box1_y1 = box1['y_center'] - box1['height'] / 2
    box1_x2 = box1['x_center'] + box1['width'] / 2
    box1_y2 = box1['y_center'] + box1['height'] / 2

    box2_x1 = box2['x_center'] - box2['width'] / 2
    box2_y1 = box2['y_center'] - box2['height'] / 2
    box2_x2 = box2['x_center'] + box2['width'] / 2
    box2_y2 = box2['y_center'] + box2['height'] / 2

    # Calculate intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou


def calculate_center_distance(box1, box2):
    """
    Calculate Euclidean distance between centers of two boxes
    """
    x1, y1 = box1['x_center'], box1['y_center']
    x2, y2 = box2['x_center'], box2['y_center']
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calculate_iou_DF(results_df: pd.DataFrame, gt_df: pd.DataFrame):
    IoU_data = []

    for (_, result_row), (_, gt_row) in zip(results_df.iterrows(), gt_df.iterrows()):
        IoU = calculate_iou(result_row, gt_row)
        IoU_data.append(IoU)

    return IoU_data


def plot_data_IoU(to_plot: pd.DataFrame, output_dir: Path):
    sns.set_style("darkgrid")
    file_output = output_dir.joinpath("IoU_residuals.png")

    fig, axes = plt.subplots(2, len(to_plot["Model"].unique()), figsize=(15, 10))

    exclude_zero = to_plot.copy()
    exclude_zero = exclude_zero[exclude_zero["IoU"] != 0]
    for i, data in enumerate([exclude_zero, to_plot]):

        for j, (model_version, model_data) in enumerate(data.groupby("Model")):
            ax = axes[i, j]
            ax.hist(model_data["IoU"], bins=100, alpha=0.6)
            ax.set_title(model_version)
            ax.set_xlabel("IoU Residuals (n=0.01)")
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(file_output)


def plot_euclidean(to_plot: pd.DataFrame, output_dir: Path):
    sns.set_style("darkgrid")
    file_output = output_dir.joinpath("euc_residuals.png")
    fig, axes = plt.subplots(1, len(to_plot["Model"].unique()), figsize=(15, 10))
    for i, (model_version, model_data) in enumerate(to_plot.groupby("Model")):
        ax = axes[i]
        ax.hist(model_data["euc_dist"], bins=100, alpha=0.6)
        ax.set_title(model_version)
        ax.set_xlabel("Euclidean distance(n=0.01)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_yscale("log")



    plt.tight_layout()
    plt.savefig(file_output)


def main():
    models = ["3", "5", "8"]
    matched_csv_dir = Path("./data/matched_csv")
    output_dir = Path("./results/residuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = load_data(matched_csv_dir, models, "_matched.csv")
    truth_data = load_data(matched_csv_dir, models, "_truth_matched.csv")

    IoU_data = results_data[["Model", "dataset", "img_id", "confidence"]].copy()
    IoU_data["IoU"] = calculate_iou_DF(results_data, truth_data)

    euclidian_data = results_data[["Model", "dataset", "img_id", "confidence"]].copy()
    euclidian_data["euc_dist"] = calculate_center_distance(results_data, truth_data)

    plot_data_IoU(IoU_data, output_dir)
    plot_euclidean(euclidian_data, output_dir)


if __name__ == "__main__":
    main()
