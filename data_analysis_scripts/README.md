# Data Analysis Scripts

Scripts used to analyze the results from the LISA x YOLO data

# PLEASE NOTE ALL SCRIPTS SHOULD BE RUN FROM PROJECT ROOT

### confidence_level_euc_analysis.py
Summary of function:
This script analyzes the relationship between the confidence levels of YOLO models (YOLOv3, YOLOv5, YOLOv8) and the Euclidean distance between detected bounding boxes and the corresponding ground truth boxes. The script generates scatter plots showing this relationship for each model, with trend lines using linear regression and LOESS smoothing. The analysis is performed on data loaded from CSV files containing matched detection data for YOLO models and ground truth data.

Input:
The script expects a directory containing CSV files with detection data for YOLO models and ground truth data.

Output:
The script generates and saves a PNG image containing scatter plots. 
The plots show:
-Confidence level vs. Euclidean distance for each YOLO model (YOLOv3, YOLOv5, YOLOv8).
-Trend lines representing the relationship, using both linear regression and LOESS smoothing.


### confidence_level_iou_analysis.py
Summary of function:
This script analyzes the relationship between the Intersection over Union (IoU) and the confidence levels of YOLO models (YOLOv3, YOLOv5, YOLOv8). The script calculates IoU between the predicted bounding boxes and the corresponding ground truth boxes and generates scatter plots that visualize this relationship for each model. It includes trend lines using both linear regression and LOESS smoothing. Additionally, the script computes and displays the correlation coefficient (r) for each model's IoU and confidence data, allowing for an assessment of the strength and direction of the relationship between IoU and confidence levels.

Input:
The script expects a directory containing CSV files with detection data for YOLO models and ground truth data.

Output:
The script generates and saves a PNG image containing scatter plots. 
The plots show:
-Confidence level vs. IoU for each YOLO model (YOLOv3, YOLOv5, YOLOv8).
-Trend lines representing the relationship, using both linear regression and LOESS smoothing.
-The correlation coefficient (r) for each model's IoU and confidence data, indicating the strength of the linear relationship.


### counts_histogram.py
Summary of function:
The script is designed to analyze YOLO-based detection performance across multiple model versions (YOLOv3, YOLOv5, YOLOv8) by calculating Intersection over Union (IoU) scores between predicted and ground truth bounding boxes. The script then generates several plots showing detection counts and percentages, both overall and for individual image sequences, as well as during day and night conditions. These visualizations allow for the analysis of how well different models perform, highlighting matched, missing, and misclassified detections.

Input:
-Matched Data (CSV files)
-Ground Truth Data (CSV file)


Output:
Visualization Files (PNG images):
    -overall_counts_bars.png: This plot shows the counts and percentages of matched, missing, and misclassified detections for each model version (YOLOv3, YOLOv5, YOLOv8) relative to the total ground truth.
    -Sequence Counts Bar Plots: For each dataset (sequence), separate bar plots are created to visualize the detection performance of each model (Matched, Missing, Misclassified).
    -Day and Night Counts: Additional bar plots are created for day and night sequences by filtering the dataset for day or night sequences. These plots are included in the sequence counts bar plots for day and night.
Printed Outputs:
    The script also prints the counts of matched, missing, and misclassified detections to the console for each of the combined, thresholded, and non-zero datasets.


### f1_score_nonzero.py
Summary of function:
The script is the same as f1_score.py, except a threshold of IoU value having to be above 0.00, this means that the 2 bounding boxes MUST overlap by some amount. This was set to see variations with different restrictions on data being used to calculate the f1 score. The script calculates and visualizes F1 scores, precision, and recall for three YOLO models (YOLOv3, YOLOv5, YOLOv8) across different video sequences and lighting conditions (day/night). It reads precomputed detection evaluation results, computes classification metrics, and generates comparative bar plots for overall performance and per-sequence/day-night breakdowns.

Input:
-Matched Data (CSV Files)
-Sorted CSV

Output:
Three PNG files
f1_overall.png: Overall precision, recall, and F1 score comparison.
f1_by_sequence.png: F1 scores per sequence for each model.
f1_by_day_night.png: F1 scores grouped by lighting condition (day/night) for each model.


### f1_score_thresh.py
Summary of function:
The script is the same as f1_score.py, except a threshold of IoU value having to be atleast 0.45 was set to see variations with different restrictions on data being used to calculate the f1 score. This script calculates and visualizes F1 scores, precision, and recall for three YOLO models (YOLOv3, YOLOv5, YOLOv8) across different video sequences and lighting conditions (day/night). It reads precomputed detection evaluation results, computes classification metrics, and generates comparative bar plots for overall performance and per-sequence/day-night breakdowns.

Input:
-Matched Data (CSV Files)
-Sorted CSV

Output:
Three PNG files
f1_overall.png: Overall precision, recall, and F1 score comparison.
f1_by_sequence.png: F1 scores per sequence for each model.
f1_by_day_night.png: F1 scores grouped by lighting condition (day/night) for each model.


### f1_score.py
Summary of function:
The script calculates and visualizes F1 scores, precision, and recall for three YOLO models (YOLOv3, YOLOv5, YOLOv8) across different video sequences and lighting conditions (day/night). It reads precomputed detection evaluation results, computes classification metrics, and generates comparative bar plots for overall performance and per-sequence/day-night breakdowns.

Input:
-Matched Data (CSV Files)
-Sorted CSV

Output:
Three PNG files
f1_overall.png: Overall precision, recall, and F1 score comparison.
f1_by_sequence.png: F1 scores per sequence for each model.
f1_by_day_night.png: F1 scores grouped by lighting condition (day/night) for each model.


### IoU2CSV.py
Summary of function:
Gets IoU values between predicted and ground truth bounding boxes for YOLO models and saves the results as CSV files per model.

Input:
-Matched Data (CSV files)
-Ground Truth Data (CSV file)

Output:
Per-model CSV files (3_IoU.csv, 5_IoU.csv, 8_IoU.csv) containing IoU values


### matches_residuals.py
Summary of function:
This script computes residuals between predicted and ground truth bounding boxes from YOLO models. It calculates IoU and Euclidean distance metrics, visualizes them as histograms, and saves both the plots and metric data per model version.

Input:
-Matched Data (CSV files)
-Ground Truth Data (CSV file)

Output:
CSV files per model containing sorted IoU and Euclidean distance values (IoU_data_model{model}.csv, euclidean_data_model{model}.csv)


### ttest_dvn.py
Summary of function:
This script performs t-tests to analyze the performance differences between day and night conditions for each YOLO model (YOLOv3, YOLOv5, YOLOv8). It examines three key metrics: precision, recall, and F1 score. The script generates visualizations comparing day vs night performance and indicates statistically significant differences with asterisks (*).

Input:
- F1 score metrics CSV files from f1_score.py, f1_score_thresh.py, and f1_score_nonzero.py results

Output:
- CSV files:
  - `day_night_ttest_results.csv`: T-test results for standard IoU
  - `day_night_ttest_results_thresh.csv`: T-test results for thresholded IoU (≥0.45)
  - `day_night_ttest_results_nonzero.csv`: T-test results for non-zero IoU
- Visualization files (PNG):
  - `day_night_precision_comparison.png`: Day vs night precision comparison
  - `day_night_recall_comparison.png`: Day vs night recall comparison
  - `day_night_f1_comparison.png`: Day vs night F1 score comparison
  Each plot shows horizontal bars for day and night performance with significance markers.


### ttest_model_f1.py
Summary of function:
This script performs pairwise t-tests between the three YOLO models to determine if there are statistically significant differences in their F1 scores. The analysis is performed on the thresholded IoU data (≥0.45) to ensure fair comparison. The script generates box plots showing the distribution of F1 scores for each model and indicates significant differences.

Input:
- F1 score metrics from f1_score_thresh.py results

Output:
- CSV file:
  - `model_comparison_ttest_thresh.csv`: Pairwise t-test results between models
- Visualization file:
  - `model_comparison_thresh.png`: Box plot showing F1 score distributions with significance markers


### mannwhitney_iou.py (mwu_iou.py)
Summary of function:
This script performs Mann-Whitney U tests to compare IoU distributions between YOLO models. It analyzes the data using three approaches: all individual IoU values, sequence-wise means, and sequence-wise medians. The script generates visualizations showing these distributions and marks statistically significant differences. This non-parametric test is particularly suitable for IoU analysis as it doesn't assume normal distribution.

Input:
- Matched detection CSV files from data/matched_csv/
- Ground truth matched CSV files

Output:
- CSV file:
  - `mwu_test_results.csv`: Results of Mann-Whitney U tests for all three analysis approaches
- Visualization files:
  - `model_iou_means.png`: Bar plot of mean IoU values with significance markers
  - `model_iou_medians.png`: Bar plot of median IoU values with significance markers
  - `model_iou_boxplot.png`: Box plot showing full IoU distributions with significance markers