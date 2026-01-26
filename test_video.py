import cv2
import numpy as np
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATH = 'assets/videotest11.mp4'
MODEL_PATH = 'yolov8s.pt'   # 👉 dùng s hoặc m cho xe thật

PARKING_SLOTS = [
    np.array([(73,185),(314,200),(281,562),(44,549)], np.int32),
    np.array([(445,226),(659,236),(642,578),(437,565)], np.int32),
]

SCORE_ON  = 3
SCORE_MAX = 8

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

# ================= UTILS =================
def bottom_center(bbox):
    x1,y1,x2,y2 = bbox
    return (int((x1+x2)/2), int(y2))

def point_in_slot(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# ================= INIT =================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

slot_score = [0]*len(PARKING_SLOTS)
slot_state = [0]*len(PARKING_SLOTS)  # 0: trống, 1: đúng, 2: sai

print("PARKING CHECK - REAL CAR VERSION")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    detected = [False]*len(PARKING_SLOTS)

    results = model(frame, conf=0.25, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            if cls not in VEHICLE_CLASSES:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            if (x2-x1)*(y2-y1) < 3000:
                continue

            bc = bottom_center((x1,y1,x2,y2))

            matched = False
            for i, slot in enumerate(PARKING_SLOTS):
                if point_in_slot(bc, slot):
                    detected[i] = True
                    matched = True
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.circle(frame, bc, 5, (0,255,0), -1)
                    break

            if not matched:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.circle(frame, bc, 5, (0,0,255), -1)

    # ===== TEMPORAL =====
    for i in range(len(PARKING_SLOTS)):
        if detected[i]:
            slot_score[i] = min(slot_score[i]+1, SCORE_MAX)
            if slot_score[i] >= SCORE_ON:
                slot_state[i] = 1
        else:
            slot_score[i] = max(slot_score[i]-1, 0)
            if slot_score[i] == 0:
                slot_state[i] = 0

    ok = bad = 0
    for i, slot in enumerate(PARKING_SLOTS):
        if slot_state[i] == 1:
            color=(0,255,0); txt="DUNG"; ok+=1
        else:
            color=(150,150,150); txt="TRONG"

        cv2.polylines(frame,[slot],True,color,3)
        cv2.putText(frame,txt,
            (slot[0][0],slot[0][1]-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.rectangle(frame,(0,0),(500,50),(0,0,0),-1)
    cv2.putText(frame,f"DUNG: {ok}",
        (10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow("PARKING CHECK - REAL CAR", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
