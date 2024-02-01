from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='fashion-mnist', epochs=10, imgsz=64, save_dir='/Users/seunghunjang/Desktop/YOLO_Classification/train')