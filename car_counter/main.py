from ultralytics import YOLO
import cv2
from sort import *

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../yolo_weights/yolov8l.pt")
mask = cv2.imread('mask.png')
tracker = Sort(max_age=20)

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

limits = [400, 297, 673, 297]
total_cars = []


while True:
    sucess, img = cap.read()
    imgregion = cv2.bitwise_and(img, mask)
    result = model(imgregion, stream=True)
    detections = np.empty((0,5)) # for tracker
    for r in result:
        box = r.boxes
        for b in box:
            x1,y1,x2,y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            

            conf = round(b.conf[0].item(), 2)
            cls = b.cls[0]
            # print(int(cls))
            curClass = classNames[int(cls)]
            if curClass  == "car":
                
                # detection rectangle
                # cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

                # rect info 
                # h,w = cv2.getTextSize(f'{conf} {curClass }', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                # cv2.rectangle(img, (max(0,x1), max(0,y1)), (x1+h, y1-w), (0,0,0), cv2.FILLED)
                # cv2.putText(img, f'{conf} {classNames[int(cls)]}', (max(0,x1), max(0,y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                
                currArray = np.array([[x1,y1,x2,y2,conf]])
                detections = np.vstack((detections, currArray))
    
    result_Tracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for res in result_Tracker:
        x1, y1, x2, y2, id = res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cw, ch = x2 - x1, y2 - y1

        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 1)


        h,w = cv2.getTextSize(f'{id}', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (max(0,x1), max(0,y1)), (x1+h, y1-w), (0,0,0), cv2.FILLED)
        cv2.putText(img, f'{id}', (max(0,x1), max(0,y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cx, cy = x1+cw//2, y1+ch//2
        cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[3]+15:
            if id not in total_cars:
                total_cars.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # top counter
    x1, y1 = 10,50
    h,w = cv2.getTextSize(f'Total Cars: {len(total_cars)}', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (max(0,x1), max(0,y1)), (x1+h, y1-w), (0,0,0), cv2.FILLED)
    cv2.putText(img, f'Total Cars: {len(total_cars)}', (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgregion) #mask
    cv2.waitKey(1)