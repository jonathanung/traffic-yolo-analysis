import sys
import os
import pandas as pd
import typing


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
    
    # Calculate intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

def calculate_center_distance(box1, box2):
    """
    Calculate Euclidean distance between centers of two boxes
    """
    x1, y1 = box1['x_center'], box1['y_center']
    x2, y2 = box2['x_center'], box2['y_center']
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def return_matching_csvs(model_csv: str, ground_truth_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns DataFrames for matched model predictions, matched ground truth, 
    misclassified detections, and missing ground truth boxes.
    
    First attempts IoU matching, then falls back to closest center distance for remaining boxes.
    """
    # Read and prepare dataframes
    model_df = pd.read_csv(model_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv)

    yolo_cols = ['dataset', 'img_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence']
    model_df.columns = yolo_cols
    ground_truth_df.columns = yolo_cols

    # Initialize lists to store matches and mismatches
    yolo_matches = []
    truth_matches = []
    missing_matches = []
    misclassified = []

    # Group by dataset and image_id
    for dataset in ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']:
        model_dataset = model_df[model_df['dataset'] == dataset]
        truth_dataset = ground_truth_df[ground_truth_df['dataset'] == dataset]
        
        # Process each image in the dataset
        for img_id in truth_dataset['img_id'].unique():
            current_model = model_dataset[model_dataset['img_id'] == img_id].copy()
            current_truth = truth_dataset[truth_dataset['img_id'] == img_id].copy()
            
            # Skip if no ground truth boxes
            if len(current_truth) == 0:
                if len(current_model) > 0:
                    misclassified.extend(current_model.to_dict('records'))
                continue
            
            # Track which boxes have been matched
            model_matched = set()
            truth_matched = set()
            
            # First pass: IoU matching
            for truth_idx, truth_box in current_truth.iterrows():
                best_iou = 0.5  # Minimum IoU threshold for matching
                best_model_idx = None
                
                for model_idx, model_box in current_model.iterrows():
                    if model_idx in model_matched:
                        continue
                        
                    iou = calculate_iou(truth_box, model_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_model_idx = model_idx
                
                if best_model_idx is not None: # match found
                    truth_matched.add(truth_idx)
                    model_matched.add(best_model_idx)

                    yolo_matches.append(current_model.loc[best_model_idx].to_dict())
                    truth_matches.append(truth_box.to_dict())
            
            # Second pass: Distance-based matching for remaining boxes
            remaining_truth = current_truth.loc[~current_truth.index.isin(truth_matched)]
            remaining_model = current_model.loc[~current_model.index.isin(model_matched)]
            
            if not remaining_truth.empty and not remaining_model.empty:
                for truth_idx, truth_box in remaining_truth.iterrows():
                    best_distance = float('inf')
                    best_model_idx = None
                    
                    for model_idx, model_box in remaining_model.iterrows():
                        if model_idx in model_matched:
                            continue
                            
                        distance = calculate_center_distance(truth_box, model_box)
                        if distance < best_distance:
                            best_distance = distance
                            best_model_idx = model_idx
                    
                    if best_model_idx is not None:
                        # We found a match based on distance
                        truth_matched.add(truth_idx)
                        model_matched.add(best_model_idx)
                        
                        # Add to match lists
                        yolo_matches.append(current_model.loc[best_model_idx].to_dict())
                        truth_matches.append(truth_box.to_dict())
            
            # Add remaining unmatched boxes
            unmatched_truth = current_truth.loc[~current_truth.index.isin(truth_matched)]
            missing_matches.extend(unmatched_truth.to_dict('records'))
            
            unmatched_model = current_model.loc[~current_model.index.isin(model_matched)]
            misclassified.extend(unmatched_model.to_dict('records'))
    
    # Convert lists to DataFrames
    yolo_match_df = pd.DataFrame(yolo_matches)
    truth_match_df = pd.DataFrame(truth_matches)
    missing_match_df = pd.DataFrame(missing_matches)
    misclassified_df = pd.DataFrame(misclassified)
    
    # Ensure all DataFrames have the correct column order
    for df in [yolo_match_df, truth_match_df, missing_match_df, misclassified_df]:
        if not df.empty:
            df = df[yolo_cols]
    
    return yolo_match_df, truth_match_df, misclassified_df, missing_match_df

def main(in_dir: str, out_dir: str, models: list[str]):
    """
    Process each model's predictions against ground truth and save results
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Paths
    ground_truth_csv = f"{in_dir}/lisa_processed_label.csv"
    
    # Process each model
    for model in models:
        print(f"Processing YOLOv{model}...")
        model_csv = f"{in_dir}/yolov{model}_output.csv"
        
        # Get matching results
        yolo_match_df, truth_match_df, misclassified_df, missing_match_df = return_matching_csvs(
            model_csv, ground_truth_csv
        )
        
        # Save results
        yolo_match_df.to_csv(f"{out_dir}/yolov{model}_matched.csv", index=False)
        truth_match_df.to_csv(f"{out_dir}/yolov{model}_truth_matched.csv", index=False)
        misclassified_df.to_csv(f"{out_dir}/yolov{model}_misclassified.csv", index=False)
        missing_match_df.to_csv(f"{out_dir}/yolov{model}_missing.csv", index=False)
        
        print(f"Results saved for YOLOv{model}")

if __name__ == "__main__":
    IN_PATH = "./data/sortedcsv"
    OUT_PATH = "./data/matched_csv"
    YOLO_MODELS = ["3", "5", "8"]
    main(IN_PATH, OUT_PATH, YOLO_MODELS)
