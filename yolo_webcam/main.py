from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv2.VideoCapture("..Videos/people.mp4")

model = YOLO("../yolo_weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    sucess, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        box = r.boxes
        for b in box:
            x1,y1,x2,y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

            conf = round(b.conf[0].item(), 2)
            cls = b.cls[0]
            # print(int(cls))
            h,w = cv2.getTextSize(f'{conf} {classNames[int(cls)]}', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (max(0,x1), max(0,y1)), (x1+h, y1-w), (0,0,0), cv2.FILLED)
            cv2.putText(img, f'{conf} {classNames[int(cls)]}', (max(0,x1), max(0,y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

            
    cv2.imshow("Image", img)
    cv2.waitKey(1)