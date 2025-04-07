# CMPT 353 Final Project

Clone using:
```bash
git clone https://github.com/jonathanung/traffic-yolo-analysis --recurse-submodules
```

Note: The `--recurse-submodules` flag is important if you want to run any of the training yourself; otherwise not strictly necessary.

Install weights using instructions from [MODELS.md](MODELS.md).

### Hotlinks
[MARKERS_GO_HERE](MARKER_HOWTO.MD)

[MODELS](MODELS.md)

[STRUCTURE](STRUCTURE.md)

[DATA](DATA.md)

[DATA COLLECTION SCRIPTS](data_collection_scripts/README.md)

[DATA PROCESSING SCRIPTS](data_processing_scripts/README.md)

[DATA ANALYSIS SCRIPTS](data_analysis_scripts/README.md)

## Project: Statistical Analysis of YOLO Models for Traffic Light Detection

**1. Introduction and Motivation**

The YOLO (You Only Look Once) object detection framework has evolved significantly since its introduction in 2015, with various versions offering improvements in accuracy, precision, and computational efficiency. This project focuses on analyzing three versions of YOLO (v3, v5, and v8) specifically for traffic light detection, examining their performance through statistical analysis.

As autonomous driving and advanced driver-assistance systems (ADAS) become more prevalent, reliable traffic light detection is increasingly critical. This project aims to understand how YOLO models have improved over time for this specific task, using statistical methods to evaluate their performance.

**2. Objectives**
- **Statistical Performance Analysis:**
    - Compare YOLO models (v3, v5, v8) using metrics like precision, recall, IoU, and F1-score
    - Conduct statistical tests (t-tests, Mann-Whitney U, chi-square) to evaluate performance differences
- **Environmental Analysis:**
    - Compare model performance between daytime and nighttime conditions
- **Error Analysis:**
    - Identify and analyze common failure cases and misclassifications
    - Compare error patterns across different YOLO versions

**3. Data and Methods**
- **Dataset:**
    - LISA Traffic Light Dataset
    - Focus on daySequence1, daySequence2, nightSequence1, and nightSequence2 for testing
    - dayTrain and nightTrain for model training
- **Methodology:**
    - **Data Processing:**
        - Convert annotations to YOLO format
        - Split data into training and testing sets
    - **Model Training:**
        - Train YOLOv3, v5, and v8 models using optimal weights
        - 50 epochs of training per model
    - **Performance Evaluation:**
        - Calculate evaluation metrics (IoU, F1-score)
        - Perform statistical tests to compare model performance
        - Analyze day vs. night performance differences
    - **Statistical Analysis:**
        - Chi-square tests for detection status distribution
        - T-tests for F1-score comparisons
        - Mann-Whitney U tests for precision analysis

**4. Expected Outcomes**
- Comprehensive statistical comparison of YOLO models for traffic light detection
- Understanding of performance differences between day and night conditions
- Identification of specific improvements and limitations across YOLO versions
- Statistical validation of performance differences between models

**5. Conclusion**
This project aims to provide a thorough statistical analysis of YOLO model performance in traffic light detection, focusing on the LISA dataset. Through systematic evaluation and statistical testing, we hope to quantify improvements across YOLO versions and understand their performance characteristics under different lighting conditions.
