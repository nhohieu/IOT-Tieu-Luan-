import cv2
import numpy as np

# ========================================================
# 1. NHẬP TÊN FILE ẢNH CỦA ÔNG
IMAGE_PATH = 'assets/anhbaidoxetrong.jpg' 

# 2. TỈ LỆ MUỐN GIẢM (0.5 là 50%, 0.3 là 30%...)
SCALE_FACTOR = 0.5 
# ========================================================

parking_slots = []
current_slot = []

def mouse_callback(event, x, y, flags, param):
    global current_slot, parking_slots

    if event == cv2.EVENT_LBUTTONDOWN:
        # --- XỬ LÝ QUAN TRỌNG ---
        # Vì ảnh bị thu nhỏ, nên khi click ta phải chia cho tỉ lệ 
        # để lấy lại tọa độ gốc (Real Coordinates)
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

# Đọc ảnh gốc
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"Lỗi: Không tìm thấy ảnh '{IMAGE_PATH}'")
    exit()

# --- RESIZE ẢNH ĐỂ HIỂN THỊ ---
# Resize xuống theo tỉ lệ SCALE_FACTOR
img_display = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

cv2.namedWindow("LAY TOA DO (DA THU NHO)")
cv2.setMouseCallback("LAY TOA DO (DA THU NHO)", mouse_callback)

print(f"--- ĐANG CHẠY CHẾ ĐỘ THU NHỎ {int(SCALE_FACTOR*100)}% ---")
print("Cứ vẽ bình thường, code sẽ tự tính ra tọa độ gốc cho ông.")
print("Bấm 'q' để lấy code.")

while True:
    # Copy ảnh display để vẽ (tránh vẽ đè lên ảnh gốc)
    img_show = img_display.copy()

    # Vẽ các ô đã xong
    for i, slot in enumerate(parking_slots):
        # Vì slot chứa tọa độ GỐC, nên khi vẽ lên hình nhỏ phải nhân với SCALE
        slot_scaled = (slot * SCALE_FACTOR).astype(np.int32)
        
        cv2.polylines(img_show, [slot_scaled], True, (0, 255, 0), 2)
        cv2.putText(img_show, str(i + 1), (slot_scaled[0][0], slot_scaled[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Vẽ các điểm đang chấm
    for pt in current_slot:
        # Quy đổi tọa độ gốc -> tọa độ màn hình để vẽ chấm đỏ
        pt_scaled = (int(pt[0] * SCALE_FACTOR), int(pt[1] * SCALE_FACTOR))
        cv2.circle(img_show, pt_scaled, 5, (0, 0, 255), -1)
    
    # Vẽ đường nối tạm
    if len(current_slot) > 1:
        # Quy đổi cả mảng điểm đang vẽ
        current_arr = np.array(current_slot, np.int32)
        current_scaled = (current_arr * SCALE_FACTOR).astype(np.int32)
        cv2.polylines(img_show, [current_scaled], False, (0, 255, 255), 1)

    cv2.imshow("LAY TOA DO (DA THU NHO)", img_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# --- IN KẾT QUẢ (TỌA ĐỘ GỐC) ---
print("\n" + "="*40)
print("COPY ĐOẠN NÀY DÁN VÀO MAIN (Đây là tọa độ gốc chuẩn 100%):")
print("="*40)
print("PARKING_SLOTS = [")
for slot in parking_slots:
    pts = ", ".join([str(tuple(p)) for p in slot])
    print(f"    np.array([{pts}], np.int32),")
print("]")
print("="*40 + "\n")