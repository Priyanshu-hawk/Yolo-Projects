from ultralytics import YOLO
import cv2

model = YOLO('../yolo_weights/yolov8l.pt')  # initialize
result = model("Images/3.png", show=True)  # inference
cv2.waitKey(0)