from ultralytics import YOLO
import cv2

from coco_classnames import Classnames

FRAMES_NUMBER = 10  # number of frames for processing

# load models
coco_model = YOLO('yolov8n.pt')
licence_plate_detector = YOLO('./models/license_plate_detector.pt')

cap = cv2.VideoCapture('./assets/sample.mp4')

# read frames
ret = True
frame_number = -1
vehicles = Classnames.get_vehicles()
while ret:
    ret, frame = cap.read()
    frame_number += 1

    if ret and frame_number < FRAMES_NUMBER:
        # detect vehicles
        detections = coco_model(frame)[0]
        detected_vehicles = []

        for detection in detections.boxes.data.tolist():
            *data, class_id = detection

            if int(class_id) in vehicles:
                detected_vehicles.append(data)
