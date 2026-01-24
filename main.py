import cv2
import numpy as np

parking_slots = []
current_slot = []

def mouse_callback(event, x, y, flags, param):
    global current_slot, parking_slots

    if event == cv2.EVENT_LBUTTONDOWN:
        current_slot.append((x, y))
        print(f"Chọn điểm: {x}, {y}")

        if len(current_slot) == 4:
            parking_slots.append(np.array(current_slot, np.int32))
            current_slot = []
            print(f"Hoàn thành ô số {len(parking_slots)}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if parking_slots:
            parking_slots.pop()
            print("Đã xóa ô gần nhất")

# Mở webcam, chụp 1 frame để vẽ
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Không chụp được hình")
    exit()

cv2.namedWindow("DRAW PARKING SLOTS")
cv2.setMouseCallback("DRAW PARKING SLOTS", mouse_callback)

print("HƯỚNG DẪN:")
print("- Click 4 điểm = 1 ô đậu")
print("- Chuột phải = xóa ô gần nhất")
print("- Nhấn Q khi xong")

while True:
    display = frame.copy()

    for slot in parking_slots:
        cv2.polylines(display, [slot], True, (0, 255, 0), 2)

    for pt in current_slot:
        cv2.circle(display, pt, 5, (0, 0, 255), -1)

    cv2.imshow("DRAW PARKING SLOTS", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# In tọa độ để copy
print("\n===== COPY ĐOẠN NÀY =====")
print("PARKING_SLOTS = [")
for slot in parking_slots:
    print(f"    np.array({slot.tolist()}, np.int32),")
print("]")
print("========================")