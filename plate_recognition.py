from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, read_license_plate, write_csv

from coco_classnames import Classnames

results = {}

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

cap = cv2.VideoCapture('./assets/sample.mp4')

mot_tracker = Sort()

ret = True
frame_number = -1
vehicles = Classnames.get_vehicles()

# read frames
while ret:
    ret, frame = cap.read()
    frame_number += 1

    if ret:
        results[frame_number] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detected_vehicles = []

        for detection in detections.boxes.data.tolist():
            *vehicle_data, class_id = detection

            if int(class_id) in vehicles:
                detected_vehicles.append(vehicle_data)

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detected_vehicles))

        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, license_plate_bbox_score, class_id = license_plate
            license_plate_bbox_coords = x1, y1, x2, y2

            # assign license plate to the car
            *car_bbox_coords, car_id = get_car(license_plate, track_ids)

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # license plate filtering
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_threshold = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_threshold)

            if license_plate_text is not None:
                results[frame_number][car_id] = {'car': {'bbox': car_bbox_coords},
                                                 'license_plate': {'bbox': license_plate_bbox_coords,
                                                                   'text': license_plate_text,
                                                                   'bbox_score': license_plate_bbox_score,
                                                                   'text_score': license_plate_text_score},
                                                 }


# write results
write_csv(results, './results.csv')
