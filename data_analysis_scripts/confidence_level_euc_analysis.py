import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import os
from pathlib import Path
import matplotlib.ticker as mtick
from pandas.core.interchange.dataframe_protocol import DataFrame
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess


def load_data(matched_csv_dir: Path) -> tuple[DataFrame, DataFrame]:
    """Loads matched, missing, and misclassified data for all models."""
    yolo_dfs = []
    gt_dfs = []

    models = ["3", "5", "8"]
    statuses = {'matched': 'Matched', 'truth_matched': 'Truth'}

    for model_num in models:
        model_version = f"YOLOv{model_num}"
        for status_key, status_name in statuses.items():
            file_path = matched_csv_dir / f"yolov{model_num}_{status_key}.csv"
            try:
                df = pd.read_csv(file_path)
                df['model_version'] = model_version
                if status_key == 'truth_matched':
                    gt_dfs.append(df)
                else:
                    yolo_dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: File not found - {file_path}. Skipping.")
            except pd.errors.EmptyDataError:
                 print(f"Warning: File is empty - {file_path}. Skipping.")

    if not yolo_dfs or not gt_dfs:
        print("Error: No data loaded. Cannot generate plots.")
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(yolo_dfs, ignore_index=True), pd.concat(gt_dfs, ignore_index=True)



def calculate_center_distance(box1, box2):
    """
    Calculate Euclidean distance between centers of two boxes
    """
    x1, y1 = box1['x_center'], box1['y_center']
    x2, y2 = box2['x_center'], box2['y_center']
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def plot_data(results_df: DataFrame):
    sns.set_style("darkgrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confidence Level vs Euclidean distance by Model')

    # Plot each model's data
    for i, model in enumerate(['YOLOv3', 'YOLOv5', 'YOLOv8']):
        model_data = results_df[results_df['model_version'] == model]

        # Create scatter plot
        axes[0][i].scatter(model_data['euclidean distance'], model_data['confidence'], alpha=0.05, s=5)
        axes[0][i].set_title(model)
        axes[0][i].set_xlabel('Euclidean Distance')
        axes[0][i].set_ylabel('Confidence')
        axes[0][i].grid(True)

        # Add trend line
        linear_best_fit = scipy.stats.linregress(model_data['euclidean distance'], model_data['confidence'])
        axes[0][i].plot(model_data['euclidean distance'], linear_best_fit.intercept + linear_best_fit.slope * model_data['euclidean distance'], "b-",
                        alpha=0.8)
        axes[0][i].legend(['Data', 'Linear'])

    for i, (model_version, threshold) in enumerate({'YOLOv3':0.02, 'YOLOv5': 0.014, 'YOLOv8':0.08}.items()):
        model_data = results_df[results_df['model_version'] == model_version]
        model_data = model_data[model_data['euclidean distance'] > threshold]

        # Create scatter plot
        axes[1][i].scatter(model_data['euclidean distance'], model_data['confidence'], alpha=0.05, s=5)
        axes[1][i].set_title(model_version + f' (euclidean distance > {threshold})')
        axes[1][i].set_xlabel('Euclidean Distance')
        axes[1][i].set_ylabel('Confidence')
        axes[1][i].grid(True)

        # Add trend line
        lowess_fit = lowess(model_data['confidence'], model_data['euclidean distance'], frac=0.2)
        linear_best_fit = scipy.stats.linregress(model_data['euclidean distance'], model_data['confidence'])
        axes[1][i].plot(lowess_fit[:, 0], lowess_fit[:, 1], "r--", alpha=0.8)
        axes[1][i].plot(model_data['euclidean distance'], linear_best_fit.intercept + linear_best_fit.slope * model_data['euclidean distance'], "b-",
                        alpha=0.8)
        axes[1][i].legend(['Data', 'Lowess', 'Linear'])

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/confidence_vs_euclidean_distance.png')
    plt.close()

def main():
    matched_csv_dir = Path("data/matched_csv")
    yolo_df, ground_truth_df = load_data(matched_csv_dir)
    yolo_df["euclidean distance"] = calculate_center_distance(yolo_df, ground_truth_df)
    plot_data(yolo_df)


if __name__ == "__main__":
    main()