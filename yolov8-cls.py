from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
#model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights
model = YOLO('yolov8m-cls.pt')  # load a pretrained model (recommended for training)

save_dir = '/Users/seunghunjang/Desktop/YOLO_Classification/Train_Dir'

# Train the model
results = model.train(data='/Users/seunghunjang/Desktop/YOLO_Classification/Train_dataset', epochs=10, imgsz=64, save_dir=save_dir)