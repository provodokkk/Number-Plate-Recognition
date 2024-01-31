import numpy as np
from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car

from coco_classnames import Classnames

FRAMES_NUMBER = 10  # number of frames for processing

# load models
coco_model = YOLO('yolov8n.pt')
licence_plate_detector = YOLO('./models/license_plate_detector.pt')

cap = cv2.VideoCapture('./assets/sample.mp4')

mot_tracker = Sort()

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
            *vehicle_data, class_id = detection

            if int(class_id) in vehicles:
                detected_vehicles.append(vehicle_data)

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detected_vehicles))

        licence_plates = licence_plate_detector(frame)[0]

        for licence_plate in licence_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = licence_plate

            # assign licence plate to the car
            car_data = get_car(licence_plate, track_ids)

            # crop licence plate
            licence_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # licence plate filtering
            licence_plate_crop_gray = cv2.cvtColor(licence_plate_crop, cv2.COLOR_BGR2GRAY)
            _, licence_plate_crop_threshold = cv2.threshold(licence_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            cv2.imshow('licence_plate_crop', licence_plate_crop)
            cv2.imshow('licence_plate_crop_threshold', licence_plate_crop_threshold)

            cv2.waitKey(0)
