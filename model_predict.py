from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')

# Predict with a model
results = model('predict/path/to/img.jpg')