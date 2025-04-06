import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from pathlib import Path
from scipy import stats

output_format = '''
Data: {type}
    p-value for all: {all}
    p-value for v3 and v5: {v3_v5}
    p-value for v3 and v8: {v3_v8}
    p-value for v5 and v8: {v5_v8}
    '''


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


def load_data(matched_csv_dir: Path, THRESH: float = -1) -> pd.DataFrame:
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
                if status_key == 'matched':
                    truth_path = matched_csv_dir / f"yolov{model_num}_truth_matched.csv"
                    truth_df = pd.read_csv(truth_path)
                    df['IoU'] = pd.DataFrame([calculate_iou(yolo_row, gt_row)
                                              for yolo_row, gt_row in zip(df.to_dict('records'),
                                                                          truth_df.to_dict('records'))])
                    df['status'] = df['IoU'].apply(lambda x: 'Matched' if x > THRESH else 'Misclassified')
                else:
                    df['IoU'] = -1
                    df['status'] = status_name
                all_dfs.append(df[['model_version', 'dataset', 'status']])
            except FileNotFoundError:
                print(f"Warning: File not found - {file_path}. Skipping.")
            except pd.errors.EmptyDataError:
                print(f"Warning: File is empty - {file_path}. Skipping.")

    if not all_dfs:
        print("Error: No data loaded. Cannot calculate F1 scores.")
        return pd.DataFrame()

    to_return = pd.concat(all_dfs, ignore_index=True)
    return pd.crosstab(index=to_return['model_version'], columns=to_return['status'])


def three_way_chi_square(data: pd.DataFrame):
    yolov3_data = data.iloc[[0]]
    yolov5_data = data.iloc[[1]]
    yolov8_data = data.iloc[[2]]

    chi2, v3_v5p, _, _ = stats.chi2_contingency(pd.concat([yolov3_data, yolov5_data]))
    chi2, v3_v8p, _, _ = stats.chi2_contingency(pd.concat([yolov3_data, yolov8_data]))
    chi2, v5_v8p, _, _ = stats.chi2_contingency(pd.concat([yolov5_data, yolov8_data]))

    return [v3_v5p, v3_v8p, v5_v8p]


def print_results(data: pd.DataFrame, dataType: str):
    _, p, _, _ = stats.chi2_contingency(data)

    pval = three_way_chi_square(data)
    print(output_format.format(type=dataType, all=p, v3_v5=pval[0], v3_v8=pval[1], v5_v8=pval[2]))


def main():
    models = ["3", "5", "8"]
    data_dir = Path("data/matched_csv")

    nonzero_data = load_data(data_dir, THRESH=0.0)
    thresh_data = load_data(data_dir, THRESH=0.45)
    data = load_data(data_dir)


    print_results(data, "All included")
    print_results(nonzero_data, "Nonzero")
    print_results(thresh_data, "Thresholded")

if __name__ == "__main__":
    main()
