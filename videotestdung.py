import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "assets/parking.mp4"
MODEL_PATH = "yolov8s.pt"

CONF_THRES = 0.25
FPS_DELAY = 30
DISPLAY_SCALE = 1

VEHICLE_CLASSES = ["car", "truck", "bus"]

PARKING_SLOTS = [
    np.array([(np.int32(1077), np.int32(264)), (np.int32(1168), np.int32(377)), (np.int32(1263), np.int32(373)), (np.int32(1183), np.int32(261))], np.int32),
]

ENTER_RATIO = 0.10   
EXIT_RATIO  = 0.03   
EXIT_FRAMES = 15     

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

slot_state = [0] * len(PARKING_SLOTS)
lost_counter = [0] * len(PARKING_SLOTS)

def intersection_area(poly1, poly2, shape):
    mask1 = np.zeros(shape[:2], np.uint8)
    mask2 = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask1, [poly1], 255)
    cv2.fillPoly(mask2, [poly2], 255)
    return cv2.countNonZero(cv2.bitwise_and(mask1, mask2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    slot_hit_ratio = [0.0] * len(PARKING_SLOTS)

    results = model(frame, conf=CONF_THRES, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_poly = np.array([
                (x1,y1),(x2,y1),(x2,y2),(x1,y2)
            ], np.int32)

            for i, slot in enumerate(PARKING_SLOTS):
                inter = intersection_area(slot, car_poly, frame.shape)
                area = cv2.contourArea(slot)
                ratio = inter / area if area > 0 else 0
                slot_hit_ratio[i] = max(slot_hit_ratio[i], ratio)

    for i in range(len(PARKING_SLOTS)):
        if slot_state[i] == 0:
            if slot_hit_ratio[i] > ENTER_RATIO:
                slot_state[i] = 1
                lost_counter[i] = 0
        else:
            if slot_hit_ratio[i] < EXIT_RATIO:
                lost_counter[i] += 1
                if lost_counter[i] >= EXIT_FRAMES:
                    slot_state[i] = 0
                    lost_counter[i] = 0
            else:
                lost_counter[i] = 0

    for i, slot in enumerate(PARKING_SLOTS):
        color = (0,255,0) if slot_state[i] else (0,0,255)
        label = "DO DUNG" if slot_state[i] else "TRONG"
        cv2.polylines(frame, [slot], True, color, 2)
        cv2.putText(frame, label,
            (slot[0][0], slot[0][1]-8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Giam Sat Bai Do Xe",
               cv2.resize(frame, (int(w*DISPLAY_SCALE), int(h*DISPLAY_SCALE))))

    if cv2.waitKey(FPS_DELAY) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()