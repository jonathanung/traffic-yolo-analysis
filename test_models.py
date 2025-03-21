#!/usr/bin/env python3
import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import cv2
import torch
import sys
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Test YOLO models on LISA traffic light sequence datasets')
    parser.add_argument('--model_version', type=str, choices=['v3', 'v5', 'v8'], required=True,
                        help='YOLO model version to test (v3, v5, or v8)')
    parser.add_argument('--data_path', type=str, default='data/processed/lisa/yolo',
                        help='Path to the processed LISA dataset')
    parser.add_argument('--test_sets', type=str, default='daySequence1,daySequence2,nightSequence1,nightSequence2',
                        help='Comma-separated list of test sets')
    parser.add_argument('--weights', type=str, 
                        help='Path to the trained model weights. If not specified, uses the best.pt from training')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='',
                        help='Device to run on (e.g., cuda:0 or cpu)')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save results to txt file')
    parser.add_argument('--save_imgs', action='store_true',
                        help='Save images with detections')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    return parser.parse_args()

def find_best_weights(model_version):
    """Find the best weights file from training"""
    project_root = Path(__file__).parent
    
    if model_version == 'v3':
        weights_path = project_root / 'models/yolov3/runs/train/lisa_traffic_light/weights/best.pt'
    elif model_version == 'v5':
        weights_path = project_root / 'models/yolov5/runs/train/lisa_traffic_light/weights/best.pt'
    else:  # v8
        weights_path = project_root / 'runs/train/lisa_traffic_light/weights/best.pt'
        # Try alternative paths if not found
        if not weights_path.exists():
            weights_path = project_root / 'runs/detect/lisa_traffic_light/weights/best.pt'
    
    if weights_path.exists():
        return str(weights_path)
    
    # If not found, check more locations
    if model_version == 'v3':
        alt_path = project_root / 'models/yolov3/weights/yolov3.pt'
    elif model_version == 'v5':
        alt_path = project_root / 'models/yolov5/weights/yolov5s.pt'
    else:  # v8
        alt_path = project_root / 'models/ultralytics/yolov8/weights/yolov8s.pt'
    
    if alt_path.exists():
        print(f"Warning: Best weights not found. Using pre-trained weights: {alt_path}")
        return str(alt_path)
    
    raise FileNotFoundError(f"No weights found for YOLO{model_version}")

def load_model(model_version, weights_path, device_str):
    """Load the appropriate YOLO model"""
    print(f"Loading YOLO{model_version} model from {weights_path}")
    
    # Convert device string to torch.device
    if device_str == '':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 'cpu')
    elif device_str.isdigit():  # If just a number like '0'
        device = torch.device(f'cuda:{device_str}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    
    if model_version == 'v3':
        # Add yolov3 directory to path temporarily
        sys.path.insert(0, str(Path('models/yolov3')))
        from models.experimental import attempt_load
        model = attempt_load(weights_path, device=device)
        sys.path.pop(0)  # Remove from path
        
    elif model_version == 'v5':
        # Add yolov5 directory to path temporarily
        sys.path.insert(0, str(Path('models/yolov5')))
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(weights_path, device=device)
        sys.path.pop(0)  # Remove from path
        
    else:  # v8
        from ultralytics import YOLO
        model = YOLO(weights_path)
    
    return model, device

def run_inference(model, model_version, test_set_path, img_size, conf_thresh, iou_thresh, device, save_txt, save_imgs, output_dir):
    """Run inference on a test set and save results"""
    images_dir = test_set_path / 'images'
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return None
    
    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return None
    
    print(f"Running inference on {len(image_files)} images from {test_set_path.name}")
    
    # Create output directories
    results_dir = Path(output_dir) / f"yolo{model_version}" / test_set_path.name
    os.makedirs(results_dir, exist_ok=True)  # Create main results dir
    
    txt_dir = results_dir / 'labels'
    img_dir = results_dir / 'visualization'
    
    if save_txt:
        os.makedirs(txt_dir, exist_ok=True)
    if save_imgs:
        os.makedirs(img_dir, exist_ok=True)
    
    all_predictions = []
    
    for img_file in tqdm(image_files):
        # Run inference based on model version
        if model_version in ['v3', 'v5']:
            # Common code for v3 and v5
            img = cv2.imread(str(img_file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize and prepare image
            img_resized = cv2.resize(img_rgb, (img_size, img_size))
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                if model_version == 'v3':
                    sys.path.insert(0, str(Path('models/yolov3')))
                    from utils.general import non_max_suppression
                    pred = model(img_tensor)[0]
                    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=0)  # Class 0 is traffic light
                    sys.path.pop(0)
                else:  # v5
                    sys.path.insert(0, str(Path('models/yolov5')))
                    from utils.general import non_max_suppression
                    pred = model(img_tensor)[0]
                    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=0)  # Class 0 is traffic light
                    sys.path.pop(0)
                
                detections = pred[0].cpu().numpy() if len(pred[0]) else np.empty((0, 6))
                
        else:  # v8
            # YOLOv8 uses a different API
            results = model(str(img_file), conf=conf_thresh, iou=iou_thresh, imgsz=img_size)
            img = cv2.imread(str(img_file))
            
            # Get detections safely for YOLOv8
            detections = []
            
            # Handle YOLOv8 results
            if len(results) > 0:
                boxes = results[0].boxes  # Get boxes from first image result
                
                if boxes is not None and len(boxes) > 0:
                    # Convert boxes to the format we need
                    try:
                        # Get boxes in xyxy format (x1, y1, x2, y2)
                        if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                            xyxy_boxes = boxes.xyxy.cpu().numpy()
                            
                            # Get confidences and class IDs
                            conf_values = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else np.ones(len(xyxy_boxes))
                            cls_values = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else np.zeros(len(xyxy_boxes))
                            
                            # Create detection array in the expected format
                            for i in range(len(xyxy_boxes)):
                                x1, y1, x2, y2 = xyxy_boxes[i]
                                conf = conf_values[i]
                                cls = int(cls_values[i])
                                detections.append([x1, y1, x2, y2, conf, cls])
                        
                        detections = np.array(detections) if detections else np.empty((0, 6))
                    except Exception as e:
                        print(f"Error processing YOLOv8 detections: {e}")
                        detections = np.empty((0, 6))
                else:
                    detections = np.empty((0, 6))
            else:
                detections = np.empty((0, 6))
        
        # Convert to YOLO format and save
        orig_height, orig_width = img.shape[:2]
        
        # Process each detection
        img_predictions = []
        
        # Save the predictions to file
        if save_txt:
            with open(txt_dir / f"{img_file.stem}.txt", 'w') as f:
                for det in detections:
                    if model_version in ['v3', 'v5']:
                        x1, y1, x2, y2, conf, cls = det
                    else:  # v8
                        if det.shape[1] == 6:
                            x1, y1, x2, y2, conf, cls = det
                        else:
                            x1, y1, x2, y2, conf = det
                            cls = 0  # Assume class 0 (traffic light)
                    
                    # Convert to YOLO format (relative to image size)
                    x_center = ((x1 + x2) / 2) / orig_width
                    y_center = ((y1 + y2) / 2) / orig_height
                    width = (x2 - x1) / orig_width
                    height = (y2 - y1) / orig_height
                    
                    # Save detection in YOLO format
                    f.write(f"{int(cls)} {x_center} {y_center} {width} {height} {conf}\n")
                    
                    # Add to predictions list
                    img_predictions.append({
                        'image': img_file.name,
                        'class': int(cls),
                        'confidence': float(conf),
                        'bbox': [float(x_center), float(y_center), float(width), float(height)]
                    })
        
        all_predictions.extend(img_predictions)
    
    # Save all predictions to a JSON file
    with open(results_dir / 'predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=4)
    
    print(f"Completed inference on {test_set_path.name}. Results saved to {results_dir}")
    return all_predictions

def main():
    args = parse_args()
    project_root = Path(__file__).parent
    
    # Find weights if not specified
    if not args.weights:
        args.weights = find_best_weights(args.model_version)
    
    # Load model
    model, device = load_model(args.model_version, args.weights, args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each test set
    test_sets = args.test_sets.split(',')
    all_results = {}
    
    for test_set in test_sets:
        test_set_path = Path(args.data_path) / test_set
        if not test_set_path.exists():
            print(f"Warning: Test set directory {test_set_path} not found. Skipping.")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing test set: {test_set}")
        print(f"{'='*80}")
        
        predictions = run_inference(
            model=model,
            model_version=args.model_version,
            test_set_path=test_set_path,
            img_size=args.img_size,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            device=device,
            save_txt=args.save_txt,
            save_imgs=args.save_imgs,
            output_dir=args.output_dir
        )
        
        if predictions:
            all_results[test_set] = len(predictions)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Testing completed for YOLO{args.model_version}")
    print(f"{'='*80}")
    print(f"Model weights: {args.weights}")
    
    for test_set, count in all_results.items():
        print(f"- {test_set}: {count} detections")
    
    print(f"\nResults saved to: {Path(args.output_dir) / f'yolo{args.model_version}'}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 