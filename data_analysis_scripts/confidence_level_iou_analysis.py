import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import scipy
import os
from pathlib import Path
import matplotlib.ticker as mtick
import seaborn as sns

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in YOLO format (x_center, y_center, width, height)
    """
    # Convert from center format to corner format
    box1_x1 = box1['x_center'] - box1['width']/2
    box1_y1 = box1['y_center'] - box1['height']/2
    box1_x2 = box1['x_center'] + box1['width']/2
    box1_y2 = box1['y_center'] + box1['height']/2
    
    box2_x1 = box2['x_center'] - box2['width']/2
    box2_y1 = box2['y_center'] - box2['height']/2
    box2_x2 = box2['x_center'] + box2['width']/2
    box2_y2 = box2['y_center'] + box2['height']/2
    
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

def load_data(matched_csv_dir: Path) -> pd.DataFrame:
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

    return (pd.concat(yolo_dfs, ignore_index=True), pd.concat(gt_dfs, ignore_index=True))

def main():
    matched_csv_dir = Path("data/matched_csv")
    yolo_df, ground_truth_df = load_data(matched_csv_dir)
    
    # Match rows between yolo and ground truth based on dataset and model_version
    yolo_df['iou'] = pd.DataFrame([calculate_iou(yolo_row, gt_row) 
                                  for yolo_row, gt_row in zip(yolo_df.to_dict('records'), 
                                                            ground_truth_df.to_dict('records'))])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confidence Level vs IoU by Model')

    # Plot each model's data
    for i, model in enumerate(['YOLOv3', 'YOLOv5', 'YOLOv8']):
        model_data = yolo_df[yolo_df['model_version'] == model]
        
        # Create scatter plot
        axes[0][i].scatter(model_data['iou'], model_data['confidence'], alpha=0.5)
        axes[0][i].set_title(model)
        axes[0][i].set_xlabel('IoU')
        axes[0][i].set_ylabel('Confidence')
        axes[0][i].grid(True)
        
        # Add trend line
        # lowess_fit = lowess(model_data['confidence'], model_data['iou'], frac=0.5)
        linear_best_fit = scipy.stats.linregress(model_data['iou'], model_data['confidence'])
        # axes[i].plot(lowess_fit[:, 0], lowess_fit[:, 1], "r--", alpha=0.8)
        axes[0][i].plot(model_data['iou'], linear_best_fit.intercept + linear_best_fit.slope * model_data['iou'], "b-", alpha=0.8)

    for i, model in enumerate(['YOLOv3', 'YOLOv5', 'YOLOv8']):
        model_data = yolo_df[yolo_df['model_version'] == model]
        model_data = model_data[model_data['iou'] > 0]
        
        # Create scatter plot
        axes[1][i].scatter(model_data['iou'], model_data['confidence'], alpha=0.5)
        axes[1][i].set_title(model)
        axes[1][i].set_xlabel('IoU')
        axes[1][i].set_ylabel('Confidence')
        axes[1][i].grid(True)
        
        # Add trend line
        # lowess_fit = lowess(model_data['confidence'], model_data['iou'], frac=0.5)
        linear_best_fit = scipy.stats.linregress(model_data['iou'], model_data['confidence'])
        # axes[i].plot(lowess_fit[:, 0], lowess_fit[:, 1], "r--", alpha=0.8)
        axes[1][i].plot(model_data['iou'], linear_best_fit.intercept + linear_best_fit.slope * model_data['iou'], "b-", alpha=0.8)

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/confidence_vs_iou.png')
    plt.close()

if __name__ == "__main__":
    main()