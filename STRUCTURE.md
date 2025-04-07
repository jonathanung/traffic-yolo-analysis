```
traffic-yolo-analysis/
├── README.md
├── analysis.sh
├── DATA.md
├── full_split.sh
├── LICENSE
├── MARKER_HOWTO.MD
├── MODELS.md
├── process_data.sh
├── requirements.txt
├── STRUCTURE.md
├── test_all_models.sh
├── train_all_models.sh
├── data/
│   ├── counts/
│   │   ├── verification_counts_combined.csv
│   │   ├── verification_summary.txt
│   │   └── verification_counts.csv/
│   │       ├── _SUCCESS
│   │       ├── part-00000-f5eb509e-c7a5-49b9-a78d-b1e3e9c02ba2-c000.csv
│   │       ├── ._SUCCESS.crc
│   │       └── .part-00000-f5eb509e-c7a5-49b9-a78d-b1e3e9c02ba2-c000.csv.crc
│   ├── edge_cases/
│   │   ├── v3_misclassified/
│   │   ├── v5_misclassified/
│   │   └── v8_misclassified/
│   ├── matched_csv/
│   │   ├── all_real_labels.csv
│   │   ├── yolov3_matched.csv
│   │   ├── yolov3_misclassified.csv
│   │   ├── yolov3_missing.csv
│   │   ├── yolov3_truth_matched.csv
│   │   ├── yolov5_matched.csv
│   │   ├── yolov5_misclassified.csv
│   │   ├── yolov5_missing.csv
│   │   ├── yolov5_truth_matched.csv
│   │   ├── yolov8_matched.csv
│   │   ├── yolov8_misclassified.csv
│   │   ├── yolov8_missing.csv
│   │   └── yolov8_truth_matched.csv
│   └── sortedcsv/
│       └── lisa_processed_label.csv
├── data_analysis_scripts/
│   ├── README.md
│   ├── chi-square-test.py
│   ├── confidence_level_euc_analysis.py
│   ├── confidence_level_iou_analysis.py
│   ├── counts_histogram.py
│   ├── f1_score.py
│   ├── f1_score_nonzero.py
│   ├── f1_score_thresh.py
│   ├── IoU2CSV.py
│   ├── mannwhitney_iou.py
│   ├── matches_residuals.py
│   ├── ttest_dvn.py
│   └── ttest_model_f1.py
├── data_collection_scripts/
│   ├── README.md
│   ├── data_preprocessing.py
│   ├── download.py
│   ├── train_lisa.py
│   ├── visualize_annotations.py
│   ├── yolo3_test.py
│   ├── yolo5_test.py
│   └── yolo8_test.py
├── data_processing_scripts/
│   ├── README.md
│   ├── copy_proc_lisa.sh
│   ├── format_spark_verification.py
│   ├── light_counter.py
│   ├── lisa_processed_label2csv.py
│   ├── lisa_yolo_val_label2csv.py
│   ├── matching.py
│   └── sort_label_csvs.py
├── models/
│   ├── yolov3/
│   └── yolov5/
└── results/
    ├── counts/
    ├── f1_scores/
    │   ├── f1_metrics_by_day_night.csv
    │   ├── f1_metrics_by_model.csv
    │   └── f1_metrics_by_sequence.csv
    ├── f1_scores_nonzero/
    │   ├── f1_metrics_by_day_night.csv
    │   ├── f1_metrics_by_model.csv
    │   └── f1_metrics_by_sequence.csv
    ├── f1_scores_thresh/
    │   ├── f1_metrics_by_day_night.csv
    │   ├── f1_metrics_by_model.csv
    │   └── f1_metrics_by_sequence.csv
    ├── IoU_data/
    │   ├── YoloV3_IoU.csv
    │   ├── YoloV5_IoU.csv
    │   └── YoloV8_IoU.csv
    ├── residuals/
    │   ├── euclidean_data_modelYoloV3.csv
    │   ├── euclidean_data_modelYoloV5.csv
    │   ├── euclidean_data_modelYoloV8.csv
    │   ├── IoU_data_modelYoloV3.csv
    │   ├── IoU_data_modelYoloV5.csv
    │   └── IoU_data_modelYoloV8.csv
    └── statistical_analysis/
        ├── day_night_ttest_results.csv
        ├── day_night_ttest_results_nonzero.csv
        ├── day_night_ttest_results_thresh.csv
        ├── model_comparison_ttest_thresh.csv
        └── mwu_test_results.csv

```