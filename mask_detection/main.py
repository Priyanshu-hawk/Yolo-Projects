from ultralytics import YOLO
import cv2
import numpy as np

cap = cv2.VideoCapture("http://192.168.29.195:4747/video")
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
limits = [400, 297, 673, 297]
total_cars = []


while True:
    sucess, img = cap.read()
    result = model(img, stream=True)
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
            if curClass  == "Mask" or curClass == "NO-Mask":
                
                # detection rectangle
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

                # rect info 
                # h,w = cv2.getTextSize(f'{conf} {curClass }', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                # cv2.rectangle(img, (max(0,x1), max(0,y1)), (x1+h, y1-w), (0,0,0), cv2.FILLED)
                # cv2.putText(img, f'{conf} {classNames[int(cls)]}', (max(0,x1), max(0,y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

                h,w = cv2.getTextSize(f'{curClass }', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                rx, ry = int(x1+(x2-x1)//2), int(y1+(y2-y1)//2)

                cv2.rectangle(img, (max(0,rx), max(0,ry)), (rx+h, ry-w), (0,0,0), cv2.FILLED)
                cv2.putText(img, f'{classNames[int(cls)]}', (max(0,rx), max(0,ry)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgregion) #mask
    cv2.waitKey(1)