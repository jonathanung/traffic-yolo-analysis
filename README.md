# CMPT 353 Final Project

Clone using:

```bash
git clone https://github.com/jonathanung/traffic-yolo-analysis --recurse-submodules
```

Install weights using instructions from [MODELS.md](MODELS.md).

### Hotlinks
[MODELS](MODELS.md)

[STRUCTURE](STRUCTURE.md)

[DATA](DATA.md)

[SCRIPTS](scripts/README.md)

## Project Proposal: Evaluating YOLO for Traffic Light Detection Under Varied Conditions

**1. Introduction and Motivation**

The YOLO (You Only Look Once) object detection framework is known for its speed and accuracy. It's efficient functionality has allowed for widespread integration into many fields such as autonomous vehicles, medical imaging, and surveillance/security. However, in this project, we would like to analyze and evaluate the performance of a YOLO-based model on traffic light datasets, examining how factors such as lighting, weather and occlusion can hinder or impact the model's accuracy.

This project aligns with many of the course topics discussed such as data ingestion and cleaning, exploratory data-analysis, machine learning model evaluation, and performance optimization through data science techniques. With the introduction of a real world problem, we aim to demonstrate practical applications of the course material.

**2. Objectives**
- **Quantitative Evaluation:**
    - Assess YOLO’s detection performance using metrics like precision, recall, Intersection over Union (IoU), and F1-score.
- **Environmental Analysis:**
    - Compare model performance across different environmental conditions (e.g., daytime vs. nighttime, clear vs. adverse weather).
- **Error and Bias Investigation:**
    - Identify and analyze common failure cases, including misclassifications and localization errors. Determine if these failure cases have been accounted for overtime through new models.
- **Model Enhancement (if time allows):**
    - Propose data-driven improvements (e.g., fine-tuning, augmented training data) to enhance performance in challenging scenarios.

**3. Data and Methods**
- **Datasets:**
    
    - Primary datasets will include publicly available traffic light datasets such as the LISA Traffic Light Dataset or the Bosch Small Traffic Lights Dataset.
    - If needed, additional data may be sourced or synthesized to capture more varied environmental conditions.
- **Methodology:**
    - **Data Preprocessing:**
        - Clean and format the image and annotation data to ensure consistency (e.g., normalization, resizing, splitting into training/validation/test sets).
    - **Exploratory Data Analysis (EDA):**
        - Visualize and statistically analyze the dataset to understand the distribution of traffic light images under different conditions.
    - **Baseline Implementation:**
        - Run a pre-trained YOLO model (YOLOv3, YOLOv5, and YOLOv8) on the test set to establish baseline performance.
    - **Performance Evaluation:**
        - Compute evaluation metrics and generate visualizations (e.g., detection overlays, confusion matrices, confidence levels) to assess model performance.
    - **Environmental Segmentation:**
        - Segment the test results by conditions (e.g., lighting, weather) and analyze how performance varies.
    - **Data and Error Analysis**
        - Identify recurring errors, perform root-cause analysis, and compare the results of differing models. Determine the outcomes of model improvements over the course of time through the traffic light detectors.
	- **Model Improvement(if time allows):**
		- Experiment with fine-tuning the model on condition-specific subsets or employing data augmentation techniques.

This approach leverages several techniques covered in our course, including data cleaning, statistical analysis, machine learning evaluation, and iterative model improvement.

**4. Timeline**
- **Weeks 1–2:**
    - **Data Collection & Preprocessing:** Gather the traffic light datasets, clean the data, and perform initial exploratory analysis.
- **Weeks 3–4:**
	- **Baseline Testing:** Run the pre-trained YOLO model to establish baseline detection metrics.
- **Weeks 5–6:**
    - **Detailed Data Analysis:**
        - Segment data based on environmental conditions.
        - Compute and compare evaluation metrics across different segments.
    - **Error and Bias Analysis:**
        - Identify common failure modes and determine improvement from one model to the next.
- **Weeks 7–8:**
	- **Model Refinement (if time allows)**
		- Test potential improvement strategies (e.g., fine-tuning, data augmentation).
    - **Final Experiments & Reporting:**
        - Consolidate findings and finalize experiments
        - Reports and Project Experience summaries (accomplishment statement)

**5. Expected Findings/Contributions**
- A comprehensive evaluation of the YOLO model’s performance in detecting traffic lights under varied conditions.
- Identification of specific environmental factors that impact detection accuracy.
- Recommendations for model improvements that could inform further research or practical deployment in smart city or autonomous driving contexts.
- Application of data science techniques covered in the course and how they are implemented to solve real-world problems

**6. Conclusion**
Collectively, we believe that this project aligns with the course materials while also challenging ourselves by applying class methods into persisting real-world issues. Through the systematic analyzation of YOLO's performance on traffic light datasets, we hope to evaluate it's accuracy and robustness across various versions. This is done with the purpose of providing potentially valuable insight as a guide for future improvements.