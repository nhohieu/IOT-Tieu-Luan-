import cv2
import numpy as np
from ultralytics import YOLO

PARKING_SLOTS = [
    np.array([(np.int32(295), np.int32(1465)), (np.int32(2620), np.int32(1440)), (np.int32(2655), np.int32(2605)), (np.int32(300), np.int32(2560))], np.int32),
]

IMAGE_TEST_PATH = "assets/realcar.jpg"
DISPLAY_SCALE = 0.2


def expand_polygon(poly, scale=1.03):
    center = np.mean(poly, axis=0)
    return np.int32((poly - center) * scale + center)

model = YOLO("yolov8n.pt")
frame = cv2.imread(IMAGE_TEST_PATH)
if frame is None:
    print("Không load được ảnh")
    exit()

slot_status = [0] * len(PARKING_SLOTS)


results = model(frame, conf=0.25, verbose=False)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

   
        cx = (x1 + x2) // 2
        cy = int(y1 + (y2 - y1) * 0.75)
        check_point = (cx, cy)

        cv2.circle(frame, check_point, 10, (0,255,255), -1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 3)

        for i, slot in enumerate(PARKING_SLOTS):
            slot_expanded = expand_polygon(slot)
            if cv2.pointPolygonTest(slot_expanded, check_point, False) >= 0:
                slot_status[i] = 1
                break


count_ok = 0
for i, slot in enumerate(PARKING_SLOTS):
    if slot_status[i]:
        color = (0,255,0)
        text = f"O {i+1}: DUNG"
        count_ok += 1
    else:
        color = (0,0,255)
        text = f"O {i+1}: SAI"

    cv2.polylines(frame, [slot], True, color, 4)
    cv2.putText(frame, text,
                (slot[0][0], slot[0][1]-30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

print(f"KET QUA: {count_ok} xe dung")

frame_show = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
cv2.imshow("TEST ANH - SCALE VIEW", frame_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
