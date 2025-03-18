#!/usr/bin/env python3
import os
import subprocess
import time
import argparse
from pathlib import Path

def run_training(model_version, epochs=50, batch_size=16, img_size=640, device=0):
    """Run training for a specific YOLO version"""
    print(f"\n{'='*80}")
    print(f"STARTING YOLO{model_version} TRAINING")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    cmd = [
        "python", "scripts/train_lisa.py",
        "--model_version", model_version,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--img_size", str(img_size),
        "--device", str(device)
    ]
    
    process = subprocess.run(cmd)
    
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED YOLO{model_version} TRAINING")
    print(f"Time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{'='*80}\n")
    
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description='Train all YOLO versions on LISA dataset')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs for each model')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640, 
                        help='Image size for training')
    parser.add_argument('--device', type=str, default='0', 
                        help='CUDA device (e.g., 0 or 0,1)')
    parser.add_argument('--versions', type=str, default='v3,v5,v8',
                        help='Comma-separated list of YOLO versions to train')
    args = parser.parse_args()
    
    # Parse versions to train
    versions_to_train = args.versions.split(',')
    
    # Record start time for the whole process
    total_start_time = time.time()
    
    # Create a results directory if it doesn't exist
    results_dir = Path("training_results")
    results_dir.mkdir(exist_ok=True)
    
    # Run training for each specified version
    for version in versions_to_train:
        if version not in ['v3', 'v5', 'v8']:
            print(f"Warning: Unknown version {version}, skipping")
            continue
            
        # Run the training
        exit_code = run_training(
            model_version=version,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device
        )
        
        if exit_code != 0:
            print(f"Warning: YOLO{version} training exited with code {exit_code}")
    
    # Calculate and display total time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"ALL TRAINING COMPLETED")
    print(f"Total time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main() 