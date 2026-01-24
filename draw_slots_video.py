import cv2
import numpy as np

# ========================================================
# 1. ĐỔI TÊN VIDEO CỦA BẠN Ở ĐÂY
VIDEO_PATH = 'assets/videotest.mp4' 
# ========================================================

parking_slots = []
current_slot = []

def mouse_callback(event, x, y, flags, param):
    global current_slot, parking_slots

    if event == cv2.EVENT_LBUTTONDOWN:
        # Lấy trực tiếp tọa độ (x, y) không cần tính toán nhân chia gì cả
        current_slot.append((x, y))
        print(f"Click: ({x}, {y})")

        if len(current_slot) == 4:
            parking_slots.append(np.array(current_slot, np.int32))
            current_slot = []
            print(f"--> Đã xong ô số {len(parking_slots)}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if parking_slots:
            parking_slots.pop()
            print("Đã xóa ô gần nhất!")
            current_slot = []

# --- ĐỌC VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read() # Chỉ đọc đúng 1 khung hình đầu tiên
cap.release() # Đóng video ngay

if not ret:
    print(f"Lỗi: Không đọc được video '{VIDEO_PATH}'")
    exit()

# --- KHỞI TẠO CỬA SỔ ---
cv2.namedWindow("LAY TOA DO (FULL SIZE)")
cv2.setMouseCallback("LAY TOA DO (FULL SIZE)", mouse_callback)

print(f"--- CHẾ ĐỘ VẼ TỈ LỆ GỐC 1:1 ---")
print("1. Click chuột TRÁI 4 góc để vẽ ô.")
print("2. Chuột PHẢI để xóa ô nếu sai.")
print("3. Bấm phím 'q' để LẤY CODE.")

while True:
    # Copy ra để vẽ
    img_show = frame.copy()

    # Vẽ các ô đã xong
    for i, slot in enumerate(parking_slots):
        cv2.polylines(img_show, [slot], True, (0, 255, 0), 2)
        cv2.putText(img_show, str(i + 1), (slot[0][0], slot[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Vẽ điểm đang chấm
    for pt in current_slot:
        cv2.circle(img_show, pt, 5, (0, 0, 255), -1)
    
    # Vẽ đường nối tạm cho dễ nhìn
    if len(current_slot) > 1:
        cv2.polylines(img_show, [np.array(current_slot, np.int32)], False, (0, 255, 255), 1)

    cv2.imshow("LAY TOA DO (FULL SIZE)", img_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# --- IN KẾT QUẢ ---
print("\n" + "="*40)
print("XONG! COPY ĐOẠN NÀY DÁN VÀO FILE TEST_VIDEO.PY:")
print("="*40)
print("PARKING_SLOTS = [")
for slot in parking_slots:
    # Format chuỗi để copy paste không bị lỗi
    pts = ", ".join([str(tuple(p)) for p in slot])
    print(f"    np.array([{pts}], np.int32),")
print("]")
print("="*40 + "\n")