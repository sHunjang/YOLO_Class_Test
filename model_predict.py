from ultralytics import YOLO

test_img_source = '/Users/seunghunjang/Desktop/YOLO_Classification/background_remove_test'

# Load a model
model = YOLO('best.pt')

# Predict with a model
results = model(source=test_img_source, save=True)