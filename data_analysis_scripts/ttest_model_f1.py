import pandas as pd
from scipy import stats
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def perform_model_comparison(data_df):
    """
    Perform t-tests comparing F1 scores between models
    """
    models = ['YOLOv3', 'YOLOv5', 'YOLOv8']
    results = []
    
    # Perform t-test for each pair of models
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            scores1 = data_df[data_df['model_version'] == model1]['f1_score']
            scores2 = data_df[data_df['model_version'] == model2]['f1_score']
            
            t_stat, p_value = stats.ttest_ind(scores1, scores2)
            
            mean1 = scores1.mean()
            mean2 = scores2.mean()
            
            results.append({
                'comparison': f'{model1} vs {model2}',
                f'{model1}_mean': mean1,
                f'{model2}_mean': mean2,
                'difference': mean2 - mean1,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    return pd.DataFrame(results)

def export_results_to_csv(results_df: pd.DataFrame, output_dir: Path):
    """Export the t-test results to a CSV file."""
    if not results_df.empty:
        filename = 'model_comparison_ttest_thresh.csv'
        results_df.to_csv(output_dir / filename, index=False)
        print(f"Saved t-test results to {output_dir / filename}")

def print_analysis_results(results_df: pd.DataFrame):
    """Print formatted analysis results."""
    print("\nModel Comparison Analysis (Thresholded IoU):")
    print(results_df.to_string(index=False))
    
    print("\nDetailed Analysis:")
    for _, row in results_df.iterrows():
        model1, model2 = row['comparison'].split(' vs ')
        print(f"\n{row['comparison']}:")
        print(f"{model1} Mean F1: {row[f'{model1}_mean']:.3f}")
        print(f"{model2} Mean F1: {row[f'{model2}_mean']:.3f}")
        print(f"Difference ({model2} - {model1}): {row['difference']:.3f}")
        print(f"P-value: {row['p_value']:.4f}")
        print(f"Statistically Significant: {'Yes' if row['significant'] else 'No'}")

def plot_model_comparison(results_df: pd.DataFrame, data_df: pd.DataFrame, output_dir: Path):
    """Create a box plot comparing F1 scores between models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['YOLOv3', 'YOLOv5', 'YOLOv8']
    data = [data_df[data_df['model_version'] == model]['f1_score'] for model in models]
    
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    
    colors = ['#4472C4', '#ED7D31', '#70AD47']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add significance markers
    y_max = data_df['f1_score'].max()
    for idx, row in results_df.iterrows():
        if row['significant']:
            model1, model2 = row['comparison'].split(' vs ')
            pos1 = models.index(model1) + 1
            pos2 = models.index(model2) + 1
            ax.plot([pos1, pos2], [y_max + 0.05 * (idx + 1)] * 2, 'k-')
            ax.text((pos1 + pos2) / 2, y_max + 0.05 * (idx + 1), '*', 
                   ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Distribution by Model (Thresholded IoU)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_thresh.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved model comparison plot to {output_dir / 'model_comparison_thresh.png'}")

def main():
    output_dir = Path("./results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        input_path = Path("./results/f1_scores_thresh/f1_metrics_by_sequence.csv")
        f1_data = pd.read_csv(input_path)
        
        results_df = perform_model_comparison(f1_data)
        print_analysis_results(results_df)
        export_results_to_csv(results_df, output_dir)
        plot_model_comparison(results_df, f1_data, output_dir)
        
    except FileNotFoundError:
        print(f"Warning: {input_path} not found.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
