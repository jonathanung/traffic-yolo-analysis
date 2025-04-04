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
        print("Error: No data loaded. Cannot generate plots.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

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


def plot_overall_bars(data_df: pd.DataFrame, total_gt_count: int, output_dir: Path):
    if data_df.empty or total_gt_count == 0:
        print("Skipping overall plot due to missing data or zero ground truth count.")
        return

    overall_counts = data_df.groupby(['model_version', 'status']).size().unstack(fill_value=0)
    overall_counts = overall_counts.reindex(columns=['Matched', 'Missing', 'Misclassified'], fill_value=0)

    overall_percentages = (overall_counts / total_gt_count) * 100

    fig, ax = plt.subplots(figsize=(12, 7))
    n_models = len(overall_counts.index)
    n_status = len(overall_counts.columns)
    bar_width = 0.25
    index = np.arange(n_models)

    for i, status in enumerate(overall_counts.columns):
        counts = overall_counts[status]
        percentages = overall_percentages[status]
        bars = ax.bar(index + i * bar_width, counts, bar_width, label=status, color=STATUS_COLORS.get(status, 'gray'))

        for bar, perc in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                    f'{perc:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model Version')
    ax.set_ylabel('Count')
    ax.set_title('Overall Detection Counts and Percentages (Relative to Total Ground Truth)')
    ax.set_xticks(index + bar_width * (n_status - 1) / 2)
    ax.set_xticklabels(overall_counts.index)
    ax.legend(title="Status")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_counts_bars.png', dpi=300)
    plt.close(fig)
    print(f"Saved overall counts plot to {output_dir / 'overall_counts_bars.png'}")

def plot_sequence_bars(data_df: pd.DataFrame, sequence_gt_counts: dict, output_dir: Path):
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
    plt.savefig(output_dir / 'sequence_counts_bars.png', dpi=300)
    plt.close(fig)

def main():
    matched_csv_dir = Path("./data/matched_csv")
    sorted_csv_dir = Path("./data/sortedcsv")
    output_dir = Path("./results/counts")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_data = load_data(matched_csv_dir)

    sequence_gt_counts, total_gt_count = get_ground_truth_counts(sorted_csv_dir)

    plot_overall_bars(combined_data, total_gt_count, output_dir)
    plot_sequence_bars(combined_data, sequence_gt_counts, output_dir)


if __name__ == "__main__":
    main()