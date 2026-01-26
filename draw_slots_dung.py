import cv2
import numpy as np


IMAGE_PATH = 'assets/realcar.jpg' 

SCALE_FACTOR =  0.2



parking_slots = []
current_slot = []

def mouse_callback(event, x, y, flags, param):
    global current_slot, parking_slots

    if event == cv2.EVENT_LBUTTONDOWN:
      
        real_x = int(x / SCALE_FACTOR)
        real_y = int(y / SCALE_FACTOR)
        
        current_slot.append((real_x, real_y))
        print(f"Click trên hình: ({x}, {y}) -> Lưu tọa độ gốc: ({real_x}, {real_y})")

        if len(current_slot) == 4:
            parking_slots.append(np.array(current_slot, np.int32))
            current_slot = []
            print(f"--> Đã xong ô số {len(parking_slots)}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if parking_slots:
            parking_slots.pop()
            print("Đã xóa ô gần nhất!")
            current_slot = []


img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"Lỗi: Không tìm thấy ảnh '{IMAGE_PATH}'")
    exit()


img_display = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

cv2.namedWindow("LAY TOA DO (DA THU NHO)")
cv2.setMouseCallback("LAY TOA DO (DA THU NHO)", mouse_callback)


print("Bấm 'q' để thoat.")

while True:

    img_show = img_display.copy()

    
    for i, slot in enumerate(parking_slots):
       
        slot_scaled = (slot * SCALE_FACTOR).astype(np.int32)
        
        cv2.polylines(img_show, [slot_scaled], True, (0, 255, 0), 2)
        cv2.putText(img_show, str(i + 1), (slot_scaled[0][0], slot_scaled[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    for pt in current_slot:
       
        pt_scaled = (int(pt[0] * SCALE_FACTOR), int(pt[1] * SCALE_FACTOR))
        cv2.circle(img_show, pt_scaled, 5, (0, 0, 255), -1)
    
 
    if len(current_slot) > 1:
       
        current_arr = np.array(current_slot, np.int32)
        current_scaled = (current_arr * SCALE_FACTOR).astype(np.int32)
        cv2.polylines(img_show, [current_scaled], False, (0, 255, 255), 1)

    cv2.imshow("LAY TOA DO (DA THU NHO)", img_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



print("PARKING_SLOTS = [")
for slot in parking_slots:
    pts = ", ".join([str(tuple(p)) for p in slot])
    print(f"    np.array([{pts}], np.int32),")
print("]")
