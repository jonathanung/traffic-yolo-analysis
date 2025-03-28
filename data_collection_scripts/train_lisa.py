import os
import argparse
import yaml
from pathlib import Path

def main():
    # Get the project root directory (parent of the scripts directory)
    project_root = Path(__file__).parent.parent
    
    parser = argparse.ArgumentParser(description='Train a YOLO model on LISA traffic light dataset')
    parser.add_argument('--data_path', type=str, default=str(project_root / 'data/processed/lisa/yolo'), 
                        help='Path to the processed LISA dataset')
    parser.add_argument('--model_version', type=str, choices=['v3', 'v5', 'v8'], default='v8',
                        help='YOLO model version to use (v3, v5, or v8)')
    parser.add_argument('--yolov3_path', type=str, default=str(project_root / 'models/yolov3'),
                        help='Path to YOLOv3 repository')
    parser.add_argument('--yolov5_path', type=str, default=str(project_root / 'models/yolov5'),
                        help='Path to YOLOv5 repository')
    parser.add_argument('--yolov8_path', type=str, default=str(project_root / 'models/ultralytics'),
                        help='Path to Ultralytics (YOLOv8) repository')
    parser.add_argument('--weights', type=str, default=None,
                        help='Initial weights path (defaults to model-specific pretrained weights)')
    parser.add_argument('--img_size', type=int, default=640, 
                        help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs')
    parser.add_argument('--device', default='', 
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    
    # Set up model-specific paths and default weights
    if args.model_version == 'v3':
        repo_path = Path(args.yolov3_path)
        default_weights = 'yolov3.pt'
    elif args.model_version == 'v5':
        repo_path = Path(args.yolov5_path)
        default_weights = 'yolov5s.pt'
    else:  # v8
        repo_path = Path(args.yolov8_path)
        default_weights = 'yolov8s.pt'
    
    # If no weights specified, use the default for the selected model
    if args.weights is None:
        args.weights = default_weights
    
    # Print paths for debugging
    print(f"Project root: {project_root}")
    print(f"Data path: {args.data_path}")
    print(f"Repo path: {repo_path}")
    
    # Check if data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Warning: Data path {data_path} does not exist. Please run data_preprocessing.py first.")
        return
    
    # Create combined dataset YAML file from dayTrain and nightTrain
    yaml_path = create_combined_dataset(args.data_path, args.model_version)
    
    # Train the model based on selected version
    run_training(args, repo_path, yaml_path)

def create_combined_dataset(data_path, model_version):
    """Create a combined dataset from dayTrain and nightTrain with format specific to YOLO version"""
    day_path = Path(data_path) / 'dayTrain'
    night_path = Path(data_path) / 'nightTrain'
    
    # Check if the required datasets exist
    if not day_path.exists() or not night_path.exists():
        raise FileNotFoundError(f"dayTrain or nightTrain not found in {data_path}. "
                               f"Please run data_preprocessing.py first.")
    
    # Load class names from one of the YAML files
    with open(day_path / 'dayTrain.yaml', 'r') as f:
        day_config = yaml.safe_load(f)
    
    # Create version-specific YAML configuration
    if model_version == 'v8':
        # YOLOv8 format - uses a simpler format that YOLOv8 understands
        combined_config = {
            'path': str(Path(data_path).resolve()),
            'train': [str(day_path / 'images'), str(night_path / 'images')],
            'val': [str(day_path / 'images'), str(night_path / 'images')],
            'nc': day_config['nc'],
            'names': day_config['names']
        }
    else:
        # YOLOv3/v5 format
        combined_config = {
            'train': [str(day_path / 'images'), str(night_path / 'images')],
            'val': [str(day_path / 'images'), str(night_path / 'images')],
            'nc': day_config['nc'],
            'names': day_config['names']
        }
    
    # Save combined YAML
    combined_yaml_path = Path(data_path) / f'lisa_combined_{model_version}.yaml'
    with open(combined_yaml_path, 'w') as f:
        yaml.dump(combined_config, f, default_flow_style=False)
    
    print(f"Created combined dataset YAML at {combined_yaml_path}")
    return combined_yaml_path

def run_training(args, repo_path, yaml_path):
    """Run training based on selected YOLO version"""
    print(f"Starting training with YOLO{args.model_version}...")
    print(f"Using repository at: {repo_path}")
    
    # Save current directory to return to it later
    original_dir = os.getcwd()
    
    try:
        if args.model_version == 'v3':
            # YOLOv3 training
            os.chdir(repo_path)
            train_cmd = (
                f"python train.py "
                f"--data {yaml_path.resolve()} "
                f"--weights {args.weights} "
                f"--img-size {args.img_size} "
                f"--batch-size {args.batch_size} "
                f"--epochs {args.epochs} "
                f"--device {args.device} "
                f"--name lisa_traffic_light"
            )
            print(f"Running command: {train_cmd}")
            os.system(train_cmd)
            
        elif args.model_version == 'v5':
            # YOLOv5 training
            os.chdir(repo_path)
            train_cmd = (
                f"python train.py "
                f"--data {yaml_path.resolve()} "
                f"--weights {args.weights} "
                f"--img {args.img_size} "
                f"--batch {args.batch_size} "
                f"--epochs {args.epochs} "
                f"--device {args.device} "
                f"--name lisa_traffic_light"
            )
            print(f"Running command: {train_cmd}")
            os.system(train_cmd)
            
        else:  # v8
            # YOLOv8 training - can be run from any directory
            from ultralytics import YOLO
            
            # Initialize model
            if args.weights.startswith('yolov8'):
                # Use pre-trained model from ultralytics
                model = YOLO(args.weights)
            else:
                # Use custom weights file
                model = YOLO(args.weights)
            
            # Train the model
            model.train(
                data=str(yaml_path),
                imgsz=args.img_size,
                epochs=args.epochs,
                batch=args.batch_size,
                device=args.device,
                name='lisa_traffic_light'
            )
    
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    print(f"Training completed with YOLO{args.model_version}.")
    if args.model_version in ['v3', 'v5']:
        print(f"Results are stored in {repo_path}/runs/train/lisa_traffic_light/")
    else:
        print("Results are stored in runs/train/lisa_traffic_light/")

if __name__ == "__main__":
    main() 