from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')

# Predict with a model
results = model('/Users/seunghunjang/Desktop/YOLO_Classification/TOP_4_B.PNG', save=True)