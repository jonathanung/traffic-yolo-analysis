import pandas as pd
from scipy import stats
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def perform_day_night_analysis(data_df):
    """
    Perform t-tests comparing day vs night performance for each YOLO model
    using multiple metrics (precision, recall, f1_score)
    """
    models = ['YOLOv3', 'YOLOv5', 'YOLOv8']
    metrics = ['precision', 'recall', 'f1_score']
    results = []
    
    for model in models:
        model_results = {'model': model}
        
        for metric in metrics:
            # Get day and night scores for this metric
            day_scores = data_df[
                (data_df['model_version'] == model) & 
                (data_df['dataset'].str.contains('daySequence'))
            ][metric]
            
            night_scores = data_df[
                (data_df['model_version'] == model) & 
                (data_df['dataset'].str.contains('nightSequence'))
            ][metric]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(day_scores, night_scores)
            
            # Calculate means
            day_mean = day_scores.mean()
            night_mean = night_scores.mean()
            
            # Store results for this metric
            model_results.update({
                f'day_mean_{metric}': day_mean,
                f'night_mean_{metric}': night_mean,
                f'difference_{metric}': night_mean - day_mean,
                f't_statistic_{metric}': t_stat,
                f'p_value_{metric}': p_value,
                f'significant_{metric}': p_value < 0.05
            })
        
        results.append(model_results)
    
    return pd.DataFrame(results)

def export_results_to_csv(results_df: pd.DataFrame, output_dir: Path, suffix: str = ''):
    """Export the t-test results to a CSV file."""
    if not results_df.empty:
        filename = f'day_night_ttest_results{suffix}.csv'
        results_df.to_csv(output_dir / filename, index=False)
        print(f"Saved t-test results to {output_dir / filename}")

def print_analysis_results(results_df: pd.DataFrame, analysis_type: str = ''):
    """Print formatted analysis results."""
    metrics = ['precision', 'recall', 'f1_score']
    
    print(f"\nDay vs Night Performance Analysis{' - ' + analysis_type if analysis_type else ''}:")
    print(results_df.to_string(index=False))
    
    print("\nDetailed Analysis:")
    for _, row in results_df.iterrows():
        print(f"\n{row['model']} Analysis:")
        for metric in metrics:
            print(f"\n  {metric.capitalize()}:")
            print(f"    Day Mean: {row[f'day_mean_{metric}']:.3f}")
            print(f"    Night Mean: {row[f'night_mean_{metric}']:.3f}")
            print(f"    Difference (Night - Day): {row[f'difference_{metric}']:.3f}")
            print(f"    P-value: {row[f'p_value_{metric}']:.4f}")
            print(f"    Statistically Significant: {'Yes' if row[f'significant_{metric}'] else 'No'}")

def plot_day_night_comparison(results_df: pd.DataFrame, output_dir: Path, suffix: str = '', analysis_type: str = ''):
    """Create separate bar graphs for each metric comparing day and night performance."""
    metrics = ['precision', 'recall', 'f1_score']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = results_df['model'].values
        y_pos = np.arange(len(models))
        bar_width = 0.35
        
        day_bars = ax.barh(y_pos - bar_width/2, results_df[f'day_mean_{metric}'], 
                          bar_width, label='Day', color='#FDB813', alpha=0.7)
        night_bars = ax.barh(y_pos + bar_width/2, results_df[f'night_mean_{metric}'], 
                            bar_width, label='Night', color='#1B2B44', alpha=0.7)
        
        # Add significance markers
        for idx, row in results_df.iterrows():
            if row[f'significant_{metric}']:
                ax.text(max(row[f'day_mean_{metric}'], row[f'night_mean_{metric}']) + 0.02, 
                       idx, '*', fontsize=14, va='center')
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel(metric.capitalize())
        ax.set_title(f'Day vs Night {metric.capitalize()} Comparison{" - " + analysis_type if analysis_type else ""}')
        ax.legend()
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'day_night_{metric}_comparison{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {metric} day/night comparison plot to {output_dir / f'day_night_{metric}_comparison{suffix}.png'}")

def main():
    output_dir = Path("./results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        versions = [
            ('', 'f1_metrics_by_sequence.csv', 'Standard IoU'),
            ('_thresh', 'f1_metrics_by_sequence.csv', 'Thresholded IoU'),
            ('_nonzero', 'f1_metrics_by_sequence.csv', 'Non-zero IoU')
        ]
        
        for suffix, input_file, analysis_type in versions:
            try:
                input_path = Path(f"./results/f1_scores{suffix}") / input_file
                f1_data = pd.read_csv(input_path)
                
                results_df = perform_day_night_analysis(f1_data)
                print_analysis_results(results_df, analysis_type)
                export_results_to_csv(results_df, output_dir, suffix)
                
                plot_day_night_comparison(results_df, output_dir, suffix, analysis_type)
                
            except FileNotFoundError:
                print(f"Warning: {input_path} not found. Skipping {analysis_type} analysis.")
            except Exception as e:
                print(f"Error during {analysis_type} analysis: {str(e)}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
