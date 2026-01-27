import cv2
import numpy as np

# ================= CONFIG =================
VIDEO_PATH = 'assets/parking.mp4'
SCALE_FACTOR = 1   # 1.2 / 1.5 / 2.0 tuỳ màn hình

parking_slots = []
current_slot = []

# ================= MOUSE CALLBACK =================
def mouse_callback(event, x, y, flags, param):
    global current_slot, parking_slots

    # Quy đổi tọa độ từ ảnh zoom -> ảnh gốc
    orig_x = int(x / SCALE_FACTOR)
    orig_y = int(y / SCALE_FACTOR)

    if event == cv2.EVENT_LBUTTONDOWN:
        current_slot.append((orig_x, orig_y))
        print(f"Click: zoom({x},{y}) -> gốc({orig_x},{orig_y})")

        if len(current_slot) == 4:
            parking_slots.append(np.array(current_slot, np.int32))
            print(f"✔ Đã lưu ô số {len(parking_slots)}")
            current_slot = []

    elif event == cv2.EVENT_RBUTTONDOWN:
        if parking_slots:
            parking_slots.pop()
            print("✘ Đã xóa ô gần nhất")
        current_slot = []

# ================= LOAD VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Không đọc được video")
    exit()

frame_orig = frame.copy()  # Giữ frame gốc

h, w = frame_orig.shape[:2]
new_w, new_h = int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)

cv2.namedWindow("LAY TOA DO (ZOOM)")
cv2.setMouseCallback("LAY TOA DO (ZOOM)", mouse_callback)

print("HƯỚNG DẪN:")
print("- Click trái 4 điểm theo chiều kim đồng hồ để tạo 1 ô")
print("- Click phải để xóa ô gần nhất")
print("- Nhấn 'q' để kết thúc\n")

# ================= MAIN LOOP =================
while True:
    img_display = cv2.resize(frame_orig, (new_w, new_h))

    # Vẽ các ô đã lưu
    for i, slot in enumerate(parking_slots):
        zoomed_slot = (slot * SCALE_FACTOR).astype(np.int32)
        cv2.polylines(img_display, [zoomed_slot], True, (0, 255, 0), 2)
        cv2.putText(
            img_display,
            f"{i+1}",
            tuple(zoomed_slot[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Vẽ điểm đang chọn
    for pt in current_slot:
        zx, zy = int(pt[0] * SCALE_FACTOR), int(pt[1] * SCALE_FACTOR)
        cv2.circle(img_display, (zx, zy), 5, (0, 0, 255), -1)

    # Vẽ đường nối tạm
    if len(current_slot) > 1:
        temp = np.array(
            [(int(p[0]*SCALE_FACTOR), int(p[1]*SCALE_FACTOR)) for p in current_slot],
            np.int32
        )
        cv2.polylines(img_display, [temp], False, (0, 255, 255), 1)

    cv2.imshow("LAY TOA DO (ZOOM)", img_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# ================= PRINT RESULT =================
print("\n" + "="*45)
print("PARKING_SLOTS = [")
for slot in parking_slots:
    pts = ", ".join([str(tuple(p)) for p in slot])
    print(f"    np.array([{pts}], np.int32),")
print("]")
print("="*45)
