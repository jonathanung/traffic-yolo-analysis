## Note to Markers

Taking your own time to download, preprocess, train, and test the models on the pre-existing sets of data is extremely unnecessary. The necessary data for the analysis is already present in data.

All of the analysis graphs and csvs can be fetched by running:
```bash
rm -rf ./results
bash analysis.sh
```

### Project Structure
- `data/` - Contains all preprocessed data and model outputs
  - `matched_csv/` - Contains detection results for each model
  - `sortedcsv/` - Contains ground truth data
- `results/` - Will be generated after running analysis.sh, containing:
  - `f1_scores/` - F1 score analysis for standard, thresholded, and non-zero IoU
  - `statistical_analysis/` - T-test results comparing day/night performance
  - Generated visualizations and CSV files

### Analysis Scripts
- `data_analysis_scripts/`
  - `f1_score.py` - Calculates F1 scores and generates performance plots
  - `ttest_dvn.py` - Performs statistical analysis of day vs night performance
  - `counts_histogram.py` - Analyzes detection counts and distributions

### Key Output Files
After running `analysis.sh`, you'll find:
1. F1 Score Analysis:
   - `f1_metrics_by_sequence.csv` - Detailed F1 scores per sequence
   - `f1_metrics_by_model.csv` - Overall model performance
   - `f1_by_sequence.png` - Performance visualization across sequences
   - Similar files for thresholded and non-zero IoU analyses

2. Statistical Analysis:
   - `day_night_ttest_results.csv` - T-test results for day/night comparison
   - `day_night_comparison.png` - Visualization of day/night performance differences
   - Additional versions for different IoU thresholds

### Expected Results
The analysis should show:
1. Comparative performance of YOLOv3, YOLOv5, and YOLOv8
2. Statistical significance of day/night performance differences
3. Impact of different IoU thresholds on detection accuracy

### Troubleshooting
If you encounter any issues:
1. Ensure all required Python packages are installed
2. Check that the data directory structure is intact
3. Verify that all CSV files in data/ are present and not empty

### Additional Notes
- The analysis uses an IoU threshold of 0.45 for the thresholded analysis
- Statistical significance is determined at p < 0.05
- All visualizations are saved in high resolution (300 DPI)

For any questions or issues, please contact [your contact information]

