import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.ticker as mtick

# Define colors for consistency
STATUS_COLORS = {
    'Matched': 'green',
    'Missing': 'orange',
    'Misclassified': 'red'
}

THRESH = 0.45

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
    
    yolo_df = pd.concat(yolo_dfs, ignore_index=True)
    gt_df = pd.concat(gt_dfs, ignore_index=True)
    
    yolo_df['IoU'] = pd.DataFrame([calculate_iou(yolo_row, gt_row) 
                                for yolo_row, gt_row in zip(yolo_df.to_dict('records'), 
                                                        gt_df.to_dict('records'))])
    
    yolo_df_thresh = yolo_df.copy()
    yolo_df_non_zero = yolo_df.copy()
    yolo_df['status'] = 'Matched'
    yolo_df_thresh['status'] = yolo_df_thresh['IoU'].apply(lambda x: 'Matched' if x >= THRESH else 'Misclassified')
    yolo_df_non_zero['status'] = yolo_df_non_zero['IoU'].apply(lambda x: 'Matched' if x > 0 else 'Misclassified')

    yolo_dfs2 = []
    statuses2 = {'missing': 'Missing', 'misclassified': 'Misclassified'}
    for model_num in models:
        model_version = f"YOLOv{model_num}"
        for status_key, status_name in statuses2.items():
            file_path = matched_csv_dir / f"yolov{model_num}_{status_key}.csv"
            try:
                df = pd.read_csv(file_path)
                df['model_version'] = model_version
                df['status'] = status_name
                yolo_dfs2.append(df)
            except FileNotFoundError:
                print(f"Warning: File not found - {file_path}. Skipping.")
            except pd.errors.EmptyDataError:
                 print(f"Warning: File is empty - {file_path}. Skipping.")

    yolo_df2 = pd.concat(yolo_dfs2, ignore_index=True)
    yolo_df2['IoU'] = -1
    yolo_df = pd.concat([yolo_df, yolo_df2], ignore_index=True)
    yolo_df_thresh = pd.concat([yolo_df_thresh, yolo_df2], ignore_index=True)
    yolo_df_non_zero = pd.concat([yolo_df_non_zero, yolo_df2], ignore_index=True)

    return yolo_df, yolo_df_thresh, yolo_df_non_zero

def get_ground_truth_counts(sorted_csv_dir: Path) -> tuple[dict, int]:
    gt_file = sorted_csv_dir / "lisa_processed_label.csv"
    try:
        gt_df = pd.read_csv(gt_file, header=None, names=['dataset', 'img_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence'])
        sequence_counts = gt_df.groupby('dataset').size().to_dict()
        total_count = len(gt_df)
        return sequence_counts, total_count
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_file}. Cannot calculate percentages.")
        return {}, 0
    except pd.errors.EmptyDataError:
        print(f"Error: Ground truth file is empty at {gt_file}. Cannot calculate percentages.")
        return {}, 0


def plot_overall_bars(data_df: pd.DataFrame, thresh_data: pd.DataFrame, non_zero_data: pd.DataFrame, total_gt_count: int, output_dir: Path):
    if data_df.empty or total_gt_count == 0:
        print("Skipping overall plot due to missing data or zero ground truth count.")
        return

    overall_counts = data_df.groupby(['model_version', 'status']).size().unstack(fill_value=0)
    overall_counts = overall_counts.reindex(columns=['Matched', 'Missing', 'Misclassified'], fill_value=0)
    overall_percentages = (overall_counts / total_gt_count) * 100

    thresh_counts = thresh_data.groupby(['model_version', 'status']).size().unstack(fill_value=0)
    thresh_counts = thresh_counts.reindex(columns=['Matched', 'Missing', 'Misclassified'], fill_value=0)
    thresh_percentages = (thresh_counts / total_gt_count) * 100

    non_zero_counts = non_zero_data.groupby(['model_version', 'status']).size().unstack(fill_value=0)
    non_zero_counts = non_zero_counts.reindex(columns=['Matched', 'Missing', 'Misclassified'], fill_value=0)
    non_zero_percentages = (non_zero_counts / total_gt_count) * 100

    fig, ax = plt.subplots(3, 1, figsize=(12, 21))
    n_models = len(overall_counts.index)
    n_status = len(overall_counts.columns)
    bar_width = 0.25
    index = np.arange(n_models)

    for k in range(3):
        type = 'all' if k == 0 else 'thresh' if k == 1 else 'non-zero'
        for i, status in enumerate(overall_counts.columns):
            if type == 'all':
                counts = overall_counts[status]
                percentages = overall_percentages[status]
            elif type == 'thresh':
                counts = thresh_counts[status]
                percentages = thresh_percentages[status]
            elif type == 'non-zero':
                counts = non_zero_counts[status]
                percentages = non_zero_percentages[status]
            bars = ax[k].bar(index + i * bar_width, counts, bar_width, label=status, color=STATUS_COLORS.get(status, 'gray'))

            for bar, perc in zip(bars, percentages):
                height = bar.get_height()
                ax[k].text(bar.get_x() + bar.get_width() / 2., height + 5,
                        f'{perc:.1f}%',
                        ha='center', va='bottom', fontsize=9)

        ax[k].set_xlabel('Model Version')
        ax[k].set_ylabel('Count')
        ax[k].set_title(f'Overall Detection Counts and Percentages (Relative to Total Ground Truth) - {type}')
        ax[k].set_xticks(index + bar_width * (n_status - 1) / 2)
        ax[k].set_xticklabels(overall_counts.index)
        ax[k].legend(title="Status")
        ax[k].spines['top'].set_visible(False)
        ax[k].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_counts_bars.png', dpi=300)
    plt.close(fig)
    print(f"Saved overall counts plot to {output_dir / 'overall_counts_bars.png'}")

def plot_sequence_bars(data_df: pd.DataFrame, sequence_gt_counts: dict, output_dir: Path, type: str):
    if data_df.empty or not sequence_gt_counts:
        print("Skipping sequence plot due to missing data or ground truth counts.")
        return

    sequences = sorted(data_df['dataset'].unique())
    models = sorted(data_df['model_version'].unique())
    statuses = ['Matched', 'Missing', 'Misclassified']

    n_seq = len(sequences)
    n_cols = 2
    n_rows = (n_seq + n_cols - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows), sharey=True)
    axes = axes.flatten()

    fig.suptitle('Detection Counts and Percentages by Sequence and Model', fontsize=16, y=1.02)

    for i, seq in enumerate(sequences):
        ax = axes[i]
        seq_data = data_df[data_df['dataset'] == seq]
        seq_counts = seq_data.groupby(['model_version', 'status']).size().unstack(fill_value=0)
        seq_counts = seq_counts.reindex(index=models, columns=statuses, fill_value=0)

        total_gt_for_seq = sequence_gt_counts.get(seq, 0)

        if total_gt_for_seq > 0:
             seq_percentages = (seq_counts / total_gt_for_seq) * 100
        else:
             seq_percentages = pd.DataFrame(0, index=seq_counts.index, columns=seq_counts.columns)

        n_models = len(seq_counts.index)
        n_status = len(seq_counts.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        for j, status in enumerate(seq_counts.columns):
            counts = seq_counts[status]
            percentages = seq_percentages[status]
            bars = ax.bar(index + j * bar_width, counts, bar_width, label=status, color=STATUS_COLORS.get(status, 'gray'))
    
            for bar, perc in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{perc:.1f}%',
                        ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{seq} (GT: {total_gt_for_seq})')
        ax.set_xticks(index + bar_width * (n_status - 1) / 2)
        ax.set_xticklabels(seq_counts.index, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i % n_cols == 0:
            ax.set_ylabel('Count')
        if i == 0:
             ax.legend(title="Status", loc='upper right')
    
    # create plots for day and night

    for i in range(2):
        time_of_day = 'day' if i == 0 else 'night'
        ax = axes[i + n_seq]
        day_night_data = data_df[data_df['dataset'].str.contains(time_of_day)]
        day_night_counts = day_night_data.groupby(['model_version', 'status']).size().unstack(fill_value=0)
        day_night_counts = day_night_counts.reindex(index=models, columns=statuses, fill_value=0)
        
        if time_of_day == 'day':
            total_gt_for_day_night = sequence_gt_counts.get('daySequence1', 0) + sequence_gt_counts.get('daySequence2', 0)
        else:
            total_gt_for_day_night = sequence_gt_counts.get('nightSequence1', 0) + sequence_gt_counts.get('nightSequence2', 0)
        
        if total_gt_for_day_night > 0:
            day_night_percentages = (day_night_counts / total_gt_for_day_night) * 100
        else:
            day_night_percentages = pd.DataFrame(0, index=day_night_counts.index, columns=day_night_counts.columns)
            
        n_models = len(day_night_counts.index)
        n_status = len(day_night_counts.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        for j, status in enumerate(day_night_counts.columns):
            counts = day_night_counts[status]
            percentages = day_night_percentages[status]
            bars = ax.bar(index + j * bar_width, counts, bar_width, label=status, color=STATUS_COLORS.get(status, 'gray'))
            
            for bar, perc in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{perc:.1f}%',
                        ha='center', va='bottom', fontsize=8)
                
        ax.set_title(f'{"Day" if i == 0 else "Night"} (GT: {total_gt_for_day_night})')
        ax.set_xticks(index + bar_width * (n_status - 1) / 2)
        ax.set_xticklabels(day_night_counts.index, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i % n_cols == 0:
            ax.set_ylabel('Count')
        if i == 0:
            ax.legend(title="Status", loc='upper right')
    
    # After all the sequence and day/night plots are done
    total_plots_needed = n_seq + 2  # sequences + day + night
    for j in range(total_plots_needed, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_dir / f'sequence_counts_bars_{type}.png', dpi=300)
    plt.close(fig)

def main():
    matched_csv_dir = Path("./data/matched_csv")
    sorted_csv_dir = Path("./data/sortedcsv")
    output_dir = Path("./results/counts")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_data, thresh_data, non_zero_data = load_data(matched_csv_dir)

    # print counts of each dataframe's match, missing, misclassified counts
    print(combined_data.groupby(['model_version', 'status']).size().unstack(fill_value=0))
    print(thresh_data.groupby(['model_version', 'status']).size().unstack(fill_value=0))
    print(non_zero_data.groupby(['model_version', 'status']).size().unstack(fill_value=0))

    sequence_gt_counts, total_gt_count = get_ground_truth_counts(sorted_csv_dir)

    plot_overall_bars(combined_data, thresh_data, non_zero_data, total_gt_count, output_dir)
    plot_sequence_bars(combined_data, sequence_gt_counts, output_dir, "all")
    plot_sequence_bars(thresh_data, sequence_gt_counts, output_dir, "thresh")
    plot_sequence_bars(non_zero_data, sequence_gt_counts, output_dir, "non-zero")


if __name__ == "__main__":
    main()