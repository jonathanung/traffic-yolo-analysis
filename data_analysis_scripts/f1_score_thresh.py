#F1 score but only using if IoU is over 45%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import matplotlib.ticker as mtick
from typing import Tuple, Dict

# Define colors for consistency
MODEL_COLORS = {
    'YOLOv3': '#4472C4',
    'YOLOv5': '#ED7D31',
    'YOLOv8': '#70AD47'
}

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
    all_dfs = []
    models = ["3", "5", "8"]
    statuses = {'matched': 'Matched', 'missing': 'Missing', 'misclassified': 'Misclassified'}
    THRESH = 0.45

    for model_num in models:
        model_version = f"YOLOv{model_num}"
        for status_key, status_name in statuses.items():
            file_path = matched_csv_dir / f"yolov{model_num}_{status_key}.csv"
            try:
                df = pd.read_csv(file_path)
                df['model_version'] = model_version
                if status_key ==  'matched':
                    truth_path = matched_csv_dir / f"yolov{model_num}_truth_matched.csv"
                    truth_df = pd.read_csv(truth_path)
                    df['IoU'] = pd.DataFrame([calculate_iou(yolo_row, gt_row) 
                                for yolo_row, gt_row in zip(df.to_dict('records'), 
                                                        truth_df.to_dict('records'))])
                    df['status'] = df['IoU'].apply(lambda x: 'Matched' if x >= THRESH else 'Misclassified')
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

    return pd.concat(all_dfs, ignore_index=True)

def get_ground_truth_counts(sorted_csv_dir: Path) -> tuple[dict, int]:
    """Gets the ground truth counts by sequence and total."""
    gt_file = sorted_csv_dir / "lisa_processed_label.csv"
    try:
        gt_df = pd.read_csv(gt_file, header=None, 
                           names=['dataset', 'img_id', 'class_id', 'x_center', 
                                 'y_center', 'width', 'height', 'confidence'])
        sequence_counts = gt_df.groupby('dataset').size().to_dict()
        total_count = len(gt_df)
        return sequence_counts, total_count
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_file}.")
        return {}, 0
    except pd.errors.EmptyDataError:
        print(f"Error: Ground truth file is empty at {gt_file}.")
        return {}, 0
    
def calculate_f1_scores(data_df: pd.DataFrame, sequence_gt_counts: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate precision, recall, and F1 scores by model and sequence with IoU threshold."""
    if data_df.empty or not sequence_gt_counts:
        print("Cannot calculate F1 scores due to missing data.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    model_sequence_metrics = []
    model_metrics = []
    
    models = sorted(data_df['model_version'].unique())
    sequences = sorted(data_df['dataset'].unique())
    
    for model in models:
        model_data = data_df[data_df['model_version'] == model]
        
        # By sequence
        for seq in sequences:
            seq_data = model_data[model_data['dataset'] == seq]
            gt_count = sequence_gt_counts.get(seq, 0)
            if gt_count == 0:
                continue
                
            matched = seq_data[seq_data['status'] == 'Matched'].shape[0]
            misclassified = seq_data[seq_data['status'] == 'Misclassified'].shape[0]
            missing = gt_count - matched - misclassified  # Calculate missing based on GT count
            
            tp = matched
            fp = misclassified
            fn = missing
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            model_sequence_metrics.append({
                'model_version': model,
                'dataset': seq,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })
        
        # By model (overall)
        matched_total = model_data[model_data['status'] == 'Matched'].shape[0]
        misclassified_total = model_data[model_data['status'] == 'Misclassified'].shape[0]
        missing_total = sum(sequence_gt_counts.values()) - matched_total - misclassified_total
        
        tp_total = matched_total
        fp_total = misclassified_total
        fn_total = missing_total
        
        precision_total = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1_total = 2 * (precision_total * recall_total) / (precision_total + recall_total) if (precision_total + recall_total) > 0 else 0
        
        model_metrics.append({
            'model_version': model,
            'precision': precision_total,
            'recall': recall_total,
            'f1_score': f1_total,
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total
        })
    
    # Calculate metrics for day/night
    day_night_metrics = []
    for model in models:
        model_data = data_df[data_df['model_version'] == model]
        
        for time_of_day in ['day', 'night']:
            day_night_data = model_data[model_data['dataset'].str.contains(time_of_day)]
            
            if time_of_day == 'day':
                gt_count = sequence_gt_counts.get('daySequence1', 0) + sequence_gt_counts.get('daySequence2', 0)
            else:
                gt_count = sequence_gt_counts.get('nightSequence1', 0) + sequence_gt_counts.get('nightSequence2', 0)
            
            if gt_count == 0:
                continue
                
            matched = day_night_data[day_night_data['status'] == 'Matched'].shape[0]
            misclassified = day_night_data[day_night_data['status'] == 'Misclassified'].shape[0]
            missing = gt_count - matched - misclassified
            
            tp = matched
            fp = misclassified
            fn = missing
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            day_night_metrics.append({
                'model_version': model,
                'time_of_day': time_of_day.capitalize(),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })
    
    return pd.DataFrame(model_sequence_metrics), pd.DataFrame(model_metrics), pd.DataFrame(day_night_metrics)

def plot_f1_by_model(model_metrics: pd.DataFrame, output_dir: Path):
    """Plot overall F1 scores, precision, and recall by model."""
    if model_metrics.empty:
        print("Cannot generate model comparison plot due to missing data.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.25
    index = np.arange(len(model_metrics))
    
    # Plot bars for precision, recall, and f1_score
    precision_bars = ax.bar(index - bar_width, model_metrics['precision'], bar_width, 
                          label='Precision', color='#5A9BD5', alpha=0.9)
    recall_bars = ax.bar(index, model_metrics['recall'], bar_width, 
                        label='Recall', color='#ED7D31', alpha=0.9)
    f1_bars = ax.bar(index + bar_width, model_metrics['f1_score'], bar_width, 
                    label='F1 Score', color='#70AD47', alpha=0.9)
    
    # Add values on top of bars
    for bars in [precision_bars, recall_bars, f1_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_title('F1 Score, Precision, and Recall by Model')
    ax.set_xticks(index)
    ax.set_xticklabels(model_metrics['model_version'])
    ax.legend()
    ax.set_ylim(0, 1.15)  # Set y-axis limit with some padding
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_by_model.png', dpi=300)
    plt.close(fig)
    print(f"Saved F1 scores by model plot to {output_dir / 'f1_by_model.png'}")

def plot_f1_by_sequence(sequence_metrics: pd.DataFrame, output_dir: Path):
    """Plot F1 scores by sequence for each model."""
    if sequence_metrics.empty:
        print("Cannot generate sequence comparison plot due to missing data.")
        return
    
    # Group by dataset and get unique sequences
    sequences = sorted(sequence_metrics['dataset'].unique())
    models = sorted(sequence_metrics['model_version'].unique())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bar_width = 0.25
    index = np.arange(len(sequences))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        model_data = sequence_metrics[sequence_metrics['model_version'] == model]
        model_data = model_data.set_index('dataset').reindex(sequences).fillna(0)
        
        bars = ax.bar(index + (i - 1) * bar_width, model_data['f1_score'], 
                     bar_width, label=model, color=MODEL_COLORS.get(model, f'C{i}'))
        
        # Add values on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # Only show text for non-zero values
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Sequence and Model')
    ax.set_xticks(index)
    ax.set_xticklabels(sequences, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.15)  # Set y-axis limit with some padding
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_by_sequence.png', dpi=300)
    plt.close(fig)
    print(f"Saved F1 scores by sequence plot to {output_dir / 'f1_by_sequence.png'}")

def plot_f1_by_day_night(day_night_metrics: pd.DataFrame, output_dir: Path):
    """Plot F1 scores by day/night conditions for each model."""
    if day_night_metrics.empty:
        print("Cannot generate day/night comparison plot due to missing data.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    times = sorted(day_night_metrics['time_of_day'].unique())
    models = sorted(day_night_metrics['model_version'].unique())
    
    bar_width = 0.25
    index = np.arange(len(times))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        model_data = day_night_metrics[day_night_metrics['model_version'] == model]
        model_data = model_data.set_index('time_of_day').reindex(times).fillna(0)
        
        bars = ax.bar(index + (i - 1) * bar_width, model_data['f1_score'], 
                     bar_width, label=model, color=MODEL_COLORS.get(model, f'C{i}'))
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Day/Night Conditions')
    ax.set_xticks(index)
    ax.set_xticklabels(times)
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_by_day_night.png', dpi=300)
    plt.close(fig)
    print(f"Saved F1 scores by day/night plot to {output_dir / 'f1_by_day_night.png'}")

def export_metrics_to_csv(sequence_metrics: pd.DataFrame, model_metrics: pd.DataFrame, output_dir: Path):
    """Export the calculated metrics to CSV files."""
    if not sequence_metrics.empty:
        sequence_metrics.to_csv(output_dir / 'f1_metrics_by_sequence.csv', index=False)
        print(f"Saved sequence metrics to {output_dir / 'f1_metrics_by_sequence.csv'}")
    
    if not model_metrics.empty:
        model_metrics.to_csv(output_dir / 'f1_metrics_by_model.csv', index=False)
        print(f"Saved model metrics to {output_dir / 'f1_metrics_by_model.csv'}")

def main():
    matched_csv_dir = Path("./data/matched_csv")
    sorted_csv_dir = Path("./data/sortedcsv")
    output_dir = Path("./results/f1_scores_thres")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    combined_data = load_data(matched_csv_dir)
    sequence_gt_counts, total_gt_count = get_ground_truth_counts(sorted_csv_dir)
    
    # Calculate F1 scores
    sequence_metrics, model_metrics, day_night_metrics = calculate_f1_scores(combined_data, sequence_gt_counts)
    
    # Display metrics in console
    if not model_metrics.empty:
        print("\nF1 Scores by Model:")
        print(model_metrics[['model_version', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    if not sequence_metrics.empty:
        print("\nF1 Scores by Sequence and Model:")
        print(sequence_metrics[['model_version', 'dataset', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    if not day_night_metrics.empty:
        print("\nF1 Scores by Day/Night Conditions:")
        print(day_night_metrics[['model_version', 'time_of_day', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    # Plot and export results
    plot_f1_by_model(model_metrics, output_dir)
    plot_f1_by_sequence(sequence_metrics, output_dir)
    plot_f1_by_day_night(day_night_metrics, output_dir)
    
    # Export metrics to CSV (add day/night metrics)
    export_metrics_to_csv(sequence_metrics, model_metrics, output_dir)
    if not day_night_metrics.empty:
        day_night_metrics.to_csv(output_dir / 'f1_metrics_by_day_night.csv', index=False)
        print(f"Saved day/night metrics to {output_dir / 'f1_metrics_by_day_night.csv'}")

if __name__ == "__main__":
    main()