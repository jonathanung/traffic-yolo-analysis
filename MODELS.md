# Models

## YOLOv3

https://github.com/ultralytics/yolov3

```bash
mkdir -p models/yolov3/weights

python -c "import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov3/releases/download/v1.0/yolov3.pt', 'models/yolov3/weights/yolov3.pt')"
```

## YOLOv5

https://github.com/ultralytics/yolov5

```bash
mkdir -p models/yolov5/weights

python -c "import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt', 'models/yolov5/weights/yolov5.pt')"
```

## YOLOv8

https://github.com/ultralytics/ultralytics

```bash
pip install ultralytics

mkdir -p models/ultralytics/yolov8/weights

python -c "import ultralytics; print(ultralytics.__version__)"

python -c "from ultralytics import YOLO; import shutil; model = YOLO('yolov8n.pt'); shutil.copy(model.ckpt_path, 'models/ultralytics/yolov8/weights/yolov8n.pt')"
```