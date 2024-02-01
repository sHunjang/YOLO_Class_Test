from ultralytics import YOLO
import ultralytics

model = YOLO('best.pt')

source = '/Users/seunghunjang/Desktop/YOLO_Classification/Back_remove_Imgdir'

result = model.predict(source, save=True)