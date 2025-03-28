#!/usr/bin/env python3
"""
Standalone YOLOv8 testing script - completely independent of test_models.py
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import json
import numpy as np
from tqdm import tqdm

# Get project root directory (parent of scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Standard configuration - SAME FOR ALL MODELS
conf_thresh = 0.25  # Standard confidence threshold
iou_thresh = 0.45   # Standard IoU threshold for NMS

# Other configuration
data_path = PROJECT_ROOT / 'data/processed/lisa/yolo'
output_dir = PROJECT_ROOT / 'results/yolov8'
test_sets = ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']
model_weights = PROJECT_ROOT / 'runs/detect/lisa_traffic_light/weights/best.pt'
fallback_weights = PROJECT_ROOT / 'models/ultralytics/yolov8/weights/yolov8s.pt'

def main():
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLOv8 model directly
    try:
        print(f"Loading YOLOv8 model from {model_weights}")
        model = YOLO(model_weights)
        print(f"Using confidence threshold: {conf_thresh}, IoU threshold: {iou_thresh}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try fallback to standard yolov8 model
        print(f"Trying fallback to standard model: {fallback_weights}")
        if fallback_weights.exists():
            model = YOLO(fallback_weights)
        else:
            model = YOLO("yolov8s.pt")
    
    # Process each test set
    for test_set in test_sets:
        print(f"\n{'='*80}")
        print(f"Processing test set: {test_set}")
        print(f"{'='*80}")
        
        # Set up paths
        test_set_path = data_path / test_set
        images_dir = test_set_path / 'images'
        
        # Check if directory exists
        if not images_dir.exists():
            print(f"Warning: Images directory not found at {images_dir}. Skipping.")
            continue
            
        # Set up output directories
        results_dir = output_dir / test_set
        txt_dir = results_dir / 'labels'
        img_dir = results_dir / 'visualization'
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        
        # Get image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpeg'))
        image_files.sort()
        
        if not image_files:
            print(f"Error: No images found in {images_dir}")
            continue
            
        print(f"Running inference on {len(image_files)} images")
        
        # Store all predictions for JSON export
        all_predictions = []
        
        # Process images one by one to avoid any batch processing issues
        for img_file in tqdm(image_files):
            # Run prediction with YOLOv8
            results = model.predict(
                source=str(img_file),
                conf=conf_thresh,
                iou=iou_thresh,
                save=False,  # Don't save, we'll handle this ourselves
                verbose=False
            )
            
            # Get the original image for visualization
            img = cv2.imread(str(img_file))
            orig_height, orig_width = img.shape[:2]
            
            # Create a copy for drawing
            img_viz = img.copy()
            
            # Process results for this image
            if results and len(results) > 0:
                # Get the first result (only one image)
                result = results[0]
                
                # Process detections if any
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Open text file for this image
                    with open(txt_dir / f"{img_file.stem}.txt", 'w') as f:
                        # Process each detection
                        for i in range(len(boxes)):
                            try:
                                # Get box coordinates (in xyxy format)
                                xyxy = boxes.xyxy[i].cpu().numpy()
                                conf = float(boxes.conf[i].cpu().numpy())
                                cls_id = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes, 'cls') else 0
                                
                                # Unpack coordinates
                                x1, y1, x2, y2 = map(float, xyxy)
                                
                                # Draw on the visualization image
                                cv2.rectangle(img_viz, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(
                                    img_viz, f"Traffic Light {conf:.2f}", 
                                    (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                                )
                                
                                # Convert to YOLO format for text file and JSON
                                x_center = ((x1 + x2) / 2) / orig_width
                                y_center = ((y1 + y2) / 2) / orig_height
                                width = (x2 - x1) / orig_width
                                height = (y2 - y1) / orig_height
                                
                                # Write to text file
                                f.write(f"{cls_id} {x_center} {y_center} {width} {height} {conf}\n")
                                
                                # Add to predictions for JSON
                                all_predictions.append({
                                    'image': img_file.name,
                                    'class': cls_id,
                                    'confidence': conf,
                                    'bbox': [float(x_center), float(y_center), float(width), float(height)]
                                })
                            except Exception as e:
                                print(f"Error processing box {i} for image {img_file.name}: {e}")
            
            # Save visualization image
            cv2.imwrite(str(img_dir / img_file.name), img_viz)
        
        # Save all predictions to JSON
        with open(results_dir / 'predictions.json', 'w') as f:
            json.dump(all_predictions, f, indent=4)
        
        print(f"Completed inference on {test_set}. Found {len(all_predictions)} detections.")
        print(f"Results saved to {results_dir}")

    print("YOLOv8 testing completed.")

if __name__ == "__main__":
    main() 