from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
#model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights
model = YOLO('yolov8m-cls (2).pt')  # load a pretrained model (recommended for training)


# Train the model
results = model.train(data='/Users/seunghunjang/Desktop/YOLO_Classification/background_remove_dataset', epochs=30, imgsz=64)