import pandas as pd
from scipy.stats import mannwhitneyu as mwu
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
    
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union = box1_area + box2_area - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou

def load_data(model):
    df = pd.read_csv(f'data/matched_csv/{model}_matched.csv')
    truth_df = pd.read_csv(f'data/matched_csv/{model}_truth_matched.csv')
    df['IoU'] = pd.DataFrame([calculate_iou(yolo_row, gt_row) 
                            for yolo_row, gt_row in zip(df.to_dict('records'), 
                                                    truth_df.to_dict('records'))])
    df = df[df['IoU'] >= 0.45]
    return df
    

def perform_mwu_analysis(dict_of_dfs):
    """Perform Mann-Whitney U tests between models with multiple metrics"""
    models = ['yolov3', 'yolov5', 'yolov8']
    results = []
    
    # Calculate sequence-wise statistics
    sequence_stats = {}
    for model in models:
        sequence_stats[model] = {
            'means': dict_of_dfs[model].groupby('dataset')['IoU'].mean(),
            'medians': dict_of_dfs[model].groupby('dataset')['IoU'].median()
        }
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            
            stat_all, p_value_all = mwu(dict_of_dfs[model1]['IoU'], 
                                      dict_of_dfs[model2]['IoU'])
            
            stat_means, p_value_means = mwu(sequence_stats[model1]['means'],
                                          sequence_stats[model2]['means'])
            
            stat_medians, p_value_medians = mwu(sequence_stats[model1]['medians'],
                                              sequence_stats[model2]['medians'])
            
            mean1 = dict_of_dfs[model1]['IoU'].mean()
            mean2 = dict_of_dfs[model2]['IoU'].mean()
            median1 = dict_of_dfs[model1]['IoU'].median()
            median2 = dict_of_dfs[model2]['IoU'].median()
            
            results.append({
                'comparison': f'{model1} vs {model2}',
                f'{model1}_mean': mean1,
                f'{model2}_mean': mean2,
                f'{model1}_median': median1,
                f'{model2}_median': median2,
                'mean_difference': mean2 - mean1,
                'median_difference': median2 - median1,
                'all_statistic': stat_all,
                'all_p_value': p_value_all,
                'all_significant': p_value_all < 0.05,
                'means_statistic': stat_means,
                'means_p_value': p_value_means,
                'means_significant': p_value_means < 0.05,
                'medians_statistic': stat_medians,
                'medians_p_value': p_value_medians,
                'medians_significant': p_value_medians < 0.05
            })
    
    return pd.DataFrame(results)

def print_analysis_results(results_df: pd.DataFrame):
    """Print formatted analysis results"""
    print("\nMann-Whitney U Test Analysis:")
    print(results_df.to_string(index=False))
    
    print("\nDetailed Analysis:")
    for _, row in results_df.iterrows():
        model1, model2 = row['comparison'].split(' vs ')
        print(f"\n{row['comparison']}:")
        print(f"\nOverall Statistics:")
        print(f"{model1} Mean IoU: {row[f'{model1}_mean']:.3f}")
        print(f"{model2} Mean IoU: {row[f'{model2}_mean']:.3f}")
        print(f"Mean Difference ({model2} - {model1}): {row['mean_difference']:.3f}")
        print(f"{model1} Median IoU: {row[f'{model1}_median']:.3f}")
        print(f"{model2} Median IoU: {row[f'{model2}_median']:.3f}")
        print(f"Median Difference ({model2} - {model1}): {row['median_difference']:.3f}")
        
        print(f"\nMWU Test Results:")
        print(f"All data points - P-value: {row['all_p_value']:.4f} (Significant: {'Yes' if row['all_significant'] else 'No'})")
        print(f"Sequence means - P-value: {row['means_p_value']:.4f} (Significant: {'Yes' if row['means_significant'] else 'No'})")
        print(f"Sequence medians - P-value: {row['medians_p_value']:.4f} (Significant: {'Yes' if row['medians_significant'] else 'No'})")

def plot_model_comparisons(results_df: pd.DataFrame, dict_of_dfs: dict, output_dir: Path):
    """Create three plots with significance markers for each test"""
    models = ['yolov3', 'yolov5', 'yolov8']
    model_names = [m.upper() for m in models]
    colors = ['#4472C4', '#ED7D31', '#70AD47']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [dict_of_dfs[model]['IoU'].mean() for model in models]
    bars = ax.bar(model_names, means, color=colors, alpha=0.7)
    
    y_max = max(means)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    for idx, row in results_df.iterrows():
        if row['means_significant']:
            model1, model2 = row['comparison'].split(' vs ')
            pos1 = models.index(model1)
            pos2 = models.index(model2)
            ax.plot([pos1, pos2], [y_max * 1.1 + 0.02 * idx] * 2, 'k-')
            ax.text((pos1 + pos2) / 2, y_max * 1.1 + 0.02 * idx, '*', 
                   ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Mean IoU Score')
    ax.set_title('Mean IoU by Model (Thresholded ≥ 0.45)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_iou_means.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    medians = [dict_of_dfs[model]['IoU'].median() for model in models]
    bars = ax.bar(model_names, medians, color=colors, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('Median IoU Score')
    ax.set_title('Median IoU by Model (Thresholded ≥ 0.45)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_iou_medians.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [dict_of_dfs[model]['IoU'] for model in models]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add significance markers
    y_max = max(df['IoU'].max() for df in dict_of_dfs.values())
    for idx, row in results_df.iterrows():
        if row['all_significant']:
            model1, model2 = row['comparison'].split(' vs ')
            pos1 = models.index(model1) + 1
            pos2 = models.index(model2) + 1
            ax.plot([pos1, pos2], [y_max + 0.05 * (idx + 1)] * 2, 'k-')
            ax.text((pos1 + pos2) / 2, y_max + 0.05 * (idx + 1), '*', 
                   ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('IoU Score')
    ax.set_title('IoU Distribution by Model (Thresholded ≥ 0.45)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_iou_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Saved IoU comparison plots:")
    print(f"- {output_dir / 'model_iou_means.png'}")
    print(f"- {output_dir / 'model_iou_medians.png'}")
    print(f"- {output_dir / 'model_iou_boxplot.png'}")

def export_results_to_csv(results_df: pd.DataFrame, output_dir: Path):
    """Export the MWU test results to a CSV file"""
    if not results_df.empty:
        filename = 'mwu_test_results.csv'
        results_df.to_csv(output_dir / filename, index=False)
        print(f"Saved MWU test results to {output_dir / filename}")

def main():
    output_dir = Path("./results/statistical_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dict_of_dfs = {model: load_data(model) 
                      for model in ['yolov3', 'yolov5', 'yolov8']}
        
        results_df = perform_mwu_analysis(dict_of_dfs)
        
        print_analysis_results(results_df)
        export_results_to_csv(results_df, output_dir)
        plot_model_comparisons(results_df, dict_of_dfs, output_dir)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()