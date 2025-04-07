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
  - `f1_score_thresh.py` - F1 analysis with IoU threshold ≥ 0.45
  - `f1_score_nonzero.py` - F1 analysis with IoU > 0
  - `ttest_dvn.py` - Performs statistical analysis of day vs night performance
  - `ttest_model_f1.py` - Performs statistical comparison between models
  - `mannwhitney_iou.py` - Non-parametric analysis of IoU distributions
  - `counts_histogram.py` - Analyzes detection counts and distributions
  - `confidence_level_euc_analysis.py` - Analyzes confidence vs Euclidean distance
  - `confidence_level_iou_analysis.py` - Analyzes confidence vs IoU
  - `matches_residuals.py` - Computes and visualizes detection residuals
  - `IoU2CSV.py` - Processes IoU data for analysis

### Output
After running `analysis.sh`, you'll find most of your analysis in `results/`.

### Expected Results
The analysis should show:
1. Comparative performance of YOLOv3, YOLOv5, and YOLOv8
2. Statistical significance of:
   - Day/night performance differences
   - Inter-model performance differences
   - IoU distribution differences
3. Impact of different IoU thresholds on detection accuracy
4. Relationship between confidence scores and detection accuracy
5. Distribution of detection counts and errors

### Statistical Methods Used
- T-tests for comparing means (day/night and model comparisons)
- Mann-Whitney U tests for non-parametric distribution comparison
- Correlation analysis for confidence relationships
- Descriptive statistics for counts and distributions

### Troubleshooting
If you encounter any issues:
1. Ensure all required Python packages are installed
2. Check that the data directory structure is intact
3. Verify that all CSV files in data/ are present and not empty

### Additional Notes
- The analysis uses an IoU threshold of 0.45 for the thresholded analysis
- Statistical significance is determined at p < 0.05
- All visualizations are saved in high resolution (300 DPI)
- Multiple statistical approaches are used to ensure robust conclusions
- Analysis considers both parametric and non-parametric methods


