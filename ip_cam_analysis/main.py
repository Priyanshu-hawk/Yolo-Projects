from threading import Thread
import cv2, time
from ultralytics import YOLO

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/60
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)
            
    def show_frame(self):
        return self.FPS_MS, self.frame
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)

src = 'http://211.132.61.124/mjpg/video.mjpg'
threaded_camera = ThreadedCamera(src)
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
    try:
        FPS_MS, img = threaded_camera.show_frame()
        result = model(img, stream=True)

        for r in result:
            box = r.boxes
            for b in box:
                x1,y1,x2,y2 = b.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = round(b.conf[0].item(), 2)
                cls = b.cls[0]
                curClass = classNames[int(cls)]

                if curClass == "car":
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

                    # rect info 
                    h,w = cv2.getTextSize(f'{curClass }', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                    cv2.rectangle(img, (max(0,x1), max(0,y1)), (x1+h, y1-w), (0,0,0), cv2.FILLED)
                    cv2.putText(img, f'{classNames[int(cls)]}', (max(0,x1), max(0,y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)


        cv2.imshow('frame', img)
        cv2.waitKey(FPS_MS)
    except AttributeError:
        pass