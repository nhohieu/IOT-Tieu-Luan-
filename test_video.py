import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "assets/videotest4.mp4"
MODEL_PATH = "yolov8s.pt"

CONF_THRES = 0.25
FPS_DELAY = 30          # ~30 FPS
DISPLAY_SCALE = 1

# ===== NHẬN DIỆN XE =====
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]

# ===== Ô ĐỖ =====
PARKING_SLOTS = [
    np.array([(35,520), (478,553), (823,309), (614,246)], np.int32),
]

# ===== THAM SỐ =====
ENTER_RATIO = 0.10
EXIT_RATIO  = 0.03

STOP_SECONDS = 3
STOP_FRAMES  = STOP_SECONDS * FPS_DELAY
MOVE_THRESH  = 15      # pixel – nhỏ hơn là coi như đứng yên

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

slot_state   = [0] * len(PARKING_SLOTS)   # 0: an toàn | 1: đỗ trái phép
stop_counter = [0] * len(PARKING_SLOTS)
last_center  = [None] * len(PARKING_SLOTS)

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
    slot_center    = [None] * len(PARKING_SLOTS)

    results = model(frame, conf=CONF_THRES, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_poly = np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)], np.int32)
            cx, cy = (x1+x2)//2, (y1+y2)//2

            for i, slot in enumerate(PARKING_SLOTS):
                inter = intersection_area(slot, car_poly, frame.shape)
                area = cv2.contourArea(slot)
                ratio = inter / area if area > 0 else 0

                if ratio > slot_hit_ratio[i]:
                    slot_hit_ratio[i] = ratio
                    slot_center[i] = (cx, cy)

    # ===== LOGIC ĐỨNG YÊN 3 GIÂY =====
    for i in range(len(PARKING_SLOTS)):
        if slot_hit_ratio[i] > ENTER_RATIO and slot_center[i] is not None:
            if last_center[i] is not None:
                dx = slot_center[i][0] - last_center[i][0]
                dy = slot_center[i][1] - last_center[i][1]
                dist = np.sqrt(dx*dx + dy*dy)

                if dist < MOVE_THRESH:
                    stop_counter[i] += 1
                else:
                    stop_counter[i] = 0
            else:
                stop_counter[i] = 0

            last_center[i] = slot_center[i]

            if stop_counter[i] >= STOP_FRAMES:
                slot_state[i] = 1   # đỗ trái phép
        else:
            slot_state[i] = 0
            stop_counter[i] = 0
            last_center[i] = None

    # ===== VẼ =====
    for i, slot in enumerate(PARKING_SLOTS):
        if slot_state[i]:
            color = (0,0,255)
            label = "DO TRAI PHEP"
        else:
            color = (0,255,0)
            label = "AN TOAN"

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

