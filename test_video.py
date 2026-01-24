import cv2

import numpy as np

from ultralytics import YOLO



# ============================================================================

# 1. TỌA ĐỘ
PARKING_SLOTS = [
    np.array([(np.int32(95), np.int32(254)), (np.int32(276), np.int32(246)), (np.int32(269), np.int32(550)), (np.int32(105), np.int32(545))], np.int32),
    np.array([(np.int32(384), np.int32(248)), (np.int32(559), np.int32(244)), (np.int32(561), np.int32(532)), (np.int32(396), np.int32(539))], np.int32),
]

VIDEO_PATH = 'assets/videotest8.mp4'

# ============================================================================



model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)



print("--- CHẾ ĐỘ: KHÔNG HIỆN TÊN RÁC ---")



while True:

    ret, frame = cap.read()

    if not ret:

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        continue



    slot_status = [0] * len(PARKING_SLOTS)



    # Giữ conf thấp để bắt xe trắng

    results = model(frame, conf=0.1, verbose=False)



    for r in results:

        boxes = r.boxes

        for box in boxes:

            # Lấy thông tin cơ bản

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

           

            # --- LỌC ---

            # Nếu muốn không bắt cái tay thì bỏ comment dòng dưới

            # if class_name == 'person': continue



            x1, y1, x2, y2 = map(int, box.xyxy[0])

           

            # Lọc nhiễu quá nhỏ

            if (x2 - x1) * (y2 - y1) < 500:

                continue



            # --- LOGIC ---

            car_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

            cx = (x1 + x2) // 2

            cy = (y1 + y2) // 2



            for i, slot in enumerate(PARKING_SLOTS):

                if cv2.pointPolygonTest(slot, (cx, cy), False) >= 0:

                   

                    corners_inside = 0

                    for corner in car_corners:

                        if cv2.pointPolygonTest(slot, corner, False) >= 0:

                            corners_inside += 1

                   

                    if corners_inside >= 3:

                        slot_status[i] = 1

                    else:

                        slot_status[i] = 2

                   

                    # --- VẼ KHUNG (KHÔNG VIẾT TÊN) ---

                    # Chỉ vẽ khung chữ nhật để biết nó đang bắt được

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                   

                    # ĐOẠN NÀY ĐÃ XÓA LỆNH cv2.putText ĐỂ KHÔNG HIỆN TÊN NỮA

                    break



    # --- VẼ TRẠNG THÁI Ô ĐỖ ---

    count_ok = 0

    count_bad = 0

    for i, slot in enumerate(PARKING_SLOTS):

        status = slot_status[i]

        if status == 0:  

            color = (150, 150, 150); text = "TRONG"

        elif status == 1:

            color = (0, 255, 0); text = "DUNG"; count_ok += 1

        else:            

            color = (0, 0, 255); text = "SAI !!!"; count_bad += 1

           

        cv2.polylines(frame, [slot], True, color, 3)

        # Chữ trạng thái (DUNG/SAI) vẫn hiện

        cv2.putText(frame, text, (slot[0][0], slot[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)



    # Info bar

    cv2.rectangle(frame, (0,0), (500, 50), (0,0,0), -1)

    cv2.putText(frame, f"DUNG: {count_ok} | SAI: {count_bad}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)



    cv2.imshow("PARKING CHECK (NO LABELS)", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'): break



cap.release()

cv2.destroyAllWindows()