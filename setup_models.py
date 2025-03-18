#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def setup_yolov3():
    """Set up YOLOv3 model and download weights"""
    print("\n" + "="*80)
    print("Setting up YOLOv3...")
    print("="*80)
    
    # Create weights directory
    os.makedirs("models/yolov3/weights", exist_ok=True)
    
    # Download YOLOv3 weights
    weight_path = Path("models/yolov3/weights/yolov3.pt")
    if weight_path.exists():
        print(f"YOLOv3 weights already exist at {weight_path}")
    else:
        print("Downloading YOLOv3 weights...")
        download_cmd = "import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov3/releases/download/v1.0/yolov3.pt', 'models/yolov3/weights/yolov3.pt')"
        subprocess.run([sys.executable, "-c", download_cmd])
        if weight_path.exists():
            print(f"Successfully downloaded YOLOv3 weights to {weight_path}")
        else:
            print("Failed to download YOLOv3 weights")
    
    print("YOLOv3 setup complete.")

def setup_yolov5():
    """Set up YOLOv5 model and download weights"""
    print("\n" + "="*80)
    print("Setting up YOLOv5...")
    print("="*80)
    
    # Create weights directory
    os.makedirs("models/yolov5/weights", exist_ok=True)
    
    # Download YOLOv5 weights
    weight_path = Path("models/yolov5/weights/yolov5s.pt")
    if weight_path.exists():
        print(f"YOLOv5 weights already exist at {weight_path}")
    else:
        print("Downloading YOLOv5 weights...")
        download_cmd = "import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt', 'models/yolov5/weights/yolov5s.pt')"
        subprocess.run([sys.executable, "-c", download_cmd])
        if weight_path.exists():
            print(f"Successfully downloaded YOLOv5 weights to {weight_path}")
        else:
            print("Failed to download YOLOv5 weights")
    
    print("YOLOv5 setup complete.")

def setup_yolov8():
    """Set up YOLOv8 model and download weights"""
    print("\n" + "="*80)
    print("Setting up YOLOv8...")
    print("="*80)
    
    # Create weights directory
    os.makedirs("models/ultralytics/yolov8/weights", exist_ok=True)
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print(f"Ultralytics package version: {ultralytics.__version__}")
    except ImportError:
        print("Installing ultralytics package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"])
        try:
            import ultralytics
            print(f"Installed ultralytics package version: {ultralytics.__version__}")
        except ImportError:
            print("Failed to install ultralytics package")
            return
    
    # Download YOLOv8 weights
    weight_path = Path("models/ultralytics/yolov8/weights/yolov8n.pt")
    if weight_path.exists():
        print(f"YOLOv8 weights already exist at {weight_path}")
    else:
        print("Downloading YOLOv8 weights...")
        download_cmd = "from ultralytics import YOLO; import shutil; model = YOLO('yolov8n.pt'); shutil.copy(model.ckpt_path, 'models/ultralytics/yolov8/weights/yolov8n.pt')"
        subprocess.run([sys.executable, "-c", download_cmd])
        if weight_path.exists():
            print(f"Successfully downloaded YOLOv8 weights to {weight_path}")
        else:
            print("Failed to download YOLOv8 weights")
    
    print("YOLOv8 setup complete.")

def main():
    """Main function to set up all YOLO models"""
    print("Starting YOLO models setup process...")
    
    # Set up all three models
    setup_yolov3()
    setup_yolov5()
    setup_yolov8()
    
    print("\n" + "="*80)
    print("All YOLO models have been set up successfully!")
    print("="*80)

if __name__ == "__main__":
    main() 