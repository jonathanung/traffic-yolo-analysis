#!/usr/bin/env python3
"""
Standalone YOLOv3 testing script for traffic light detection
"""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import os
import json
from tqdm import tqdm

# Configuration
data_path = 'data/processed/lisa/yolo'
output_dir = 'results/yolov3'
test_sets = ['daySequence1', 'daySequence2', 'nightSequence1', 'nightSequence2']
conf_thresh = 0.25
iou_thresh = 0.45
model_weights = 'models/yolov3/runs/train/lisa_traffic_light/weights/best.pt'
fallback_weights = 'models/yolov3/weights/yolov3.pt'

def main():
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Add YOLOv3 path to sys.path temporarily
    yolov3_path = str(Path('models/yolov3').absolute())
    sys.path.insert(0, yolov3_path)
    
    # Import YOLOv3-specific modules
    try:
        from models.experimental import attempt_load
        from utils.general import non_max_suppression, scale_coords
        from utils.augmentations import letterbox
    except ImportError as e:
        print(f"Error importing YOLOv3 modules: {e}")
        print(f"Make sure the YOLOv3 repository is in models/yolov3")
        return
    
    # Load YOLOv3 model
    try:
        print(f"Loading YOLOv3 model from {model_weights}")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = attempt_load(model_weights, device=device)
        print(f"Model loaded successfully. Using device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            print(f"Trying fallback weights: {fallback_weights}")
            model = attempt_load(fallback_weights, device=device)
            print("Fallback model loaded successfully")
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            sys.path.pop(0)  # Remove YOLOv3 from path
            return
    
    # Set model to evaluation mode
    model.eval()
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # Process each test set
    for test_set in test_sets:
        print(f"\n{'='*80}")
        print(f"Processing test set: {test_set}")
        print(f"{'='*80}")
        
        # Set up paths
        test_set_path = Path(data_path) / test_set
        images_dir = test_set_path / 'images'
        
        # Check if directory exists
        if not images_dir.exists():
            print(f"Warning: Images directory not found at {images_dir}. Skipping.")
            continue
            
        # Set up output directories
        results_dir = Path(output_dir) / test_set
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
        
        # Process images one by one
        for img_file in tqdm(image_files):
            # Read and preprocess image
            img = cv2.imread(str(img_file))
            orig_img = img.copy()
            orig_height, orig_width = img.shape[:2]
            
            # Preprocess image for YOLOv3
            img_size = 640
            img = letterbox(img, img_size, stride=32)[0]
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            
            if len(img.shape) == 3:
                img = img.unsqueeze(0)  # Add batch dimension
            
            # Run inference
            with torch.no_grad():
                pred = model(img)[0]
                
                # Apply NMS
                pred = non_max_suppression(pred, conf_thresh, iou_thresh)
            
            # Process detections for this image
            img_viz = orig_img.copy()
            img_predictions = []
            
            if len(pred[0]) > 0:
                # Rescale boxes from model size to original image size
                det = pred[0].clone()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_img.shape).round()
                
                # Process each detection
                with open(txt_dir / f"{img_file.stem}.txt", 'w') as f:
                    for *xyxy, conf, cls_id in det:
                        # Convert tensor to float
                        x1, y1, x2, y2 = [float(coord) for coord in xyxy]
                        conf = float(conf)
                        cls_id = int(cls_id)
                        
                        # Draw on visualization image
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
                        img_predictions.append({
                            'image': img_file.name,
                            'class': cls_id,
                            'confidence': conf,
                            'bbox': [float(x_center), float(y_center), float(width), float(height)]
                        })
            
            # Save visualization image
            cv2.imwrite(str(img_dir / img_file.name), img_viz)
            
            # Add this image's predictions to the overall list
            all_predictions.extend(img_predictions)
        
        # Save all predictions to JSON
        with open(results_dir / 'predictions.json', 'w') as f:
            json.dump(all_predictions, f, indent=4)
        
        print(f"Completed inference on {test_set}. Found {len(all_predictions)} detections.")
        print(f"Results saved to {results_dir}")
    
    # Remove YOLOv3 from path
    sys.path.pop(0)
    
    print("YOLOv3 testing completed.")

if __name__ == "__main__":
    main() 