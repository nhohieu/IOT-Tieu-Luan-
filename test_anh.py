import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================================
# 1. DÁN TỌA ĐỘ CỦA ÔNG VÀO ĐÂY (Giữ nguyên cái cũ của ông)
PARKING_SLOTS = [
    np.array([(np.int32(486), np.int32(772)), (np.int32(826), np.int32(760)), (np.int32(832), np.int32(1350)), (np.int32(494), np.int32(1356))], np.int32),
    np.array([(np.int32(1126), np.int32(748)), (np.int32(1426), np.int32(736)), (np.int32(1470), np.int32(1276)), (np.int32(1146), np.int32(1288))], np.int32),
]

# 2. TÊN ẢNH
IMAGE_TEST_PATH = 'assets/anhcoxe.jpg' 
# ============================================================================

print("Đang chạy AI...")
model = YOLO("yolov8n.pt")
frame = cv2.imread(IMAGE_TEST_PATH)

if frame is None:
    print("Lỗi: Không tìm thấy ảnh!")
    exit()

# Tạo danh sách lưu trạng thái từng ô
# 0: Trống, 1: Đậu Đúng, 2: Đậu Sai
slot_status = [0] * len(PARKING_SLOTS)

# --- 1. CHẠY YOLO ---
results = model(frame, conf=0.1, verbose=False)

# --- 2. XỬ LÝ LOGIC ĐẬU ĐÚNG / SAI ---
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Tạo danh sách 4 góc của chiếc xe
        car_corners = [
            (x1, y1), # Góc trái trên
            (x2, y1), # Góc phải trên
            (x2, y2), # Góc phải dưới
            (x1, y2)  # Góc trái dưới
        ]
        
        # Tính tâm của xe (để biết xe đang thuộc về ô nào)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Duyệt qua từng ô đậu xe để kiểm tra
        for i, slot in enumerate(PARKING_SLOTS):
            # Bước A: Kiểm tra xem xe này có nằm ở khu vực ô 'i' không (dựa vào tâm xe)
            # Nếu tâm xe nằm trong ô -> Xe này thuộc ô này -> Bắt đầu chấm điểm
            if cv2.pointPolygonTest(slot, (cx, cy), False) >= 0:
                
                # Bước B: Đếm xem bao nhiêu góc xe nằm GỌN trong ô
                corners_inside = 0
                for corner in car_corners:
                    if cv2.pointPolygonTest(slot, corner, False) >= 0:
                        corners_inside += 1
                
                # Bước C: Phán xét
                # Nếu từ 3 góc trở lên nằm trong -> Đậu gọn -> OK
                if corners_inside >= 3:
                    slot_status[i] = 1 # ĐẬU ĐÚNG
                else:
                    # Tâm ở trong mà các góc lòi ra ngoài -> Đậu ẩu -> SAI
                    slot_status[i] = 2 # ĐẬU SAI
                
                # Vẽ khung xe và các góc để debug (nhìn cho pro)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2) # Màu Cyan
                break 

# --- 3. VẼ KẾT QUẢ RA MÀN HÌNH ---
count_ok = 0
count_bad = 0

for i, slot in enumerate(PARKING_SLOTS):
    status = slot_status[i]
    
    if status == 0:
        # TRỐNG -> Màu Trắng hoặc Xám (để đỡ nhầm với đậu đúng)
        color = (200, 200, 200) 
        text = f"O {i+1}: TRONG"
        thickness = 2
        
    elif status == 1:
        # ĐẬU ĐÚNG -> MÀU XANH LÁ
        color = (0, 255, 0)
        text = f"O {i+1}: DUNG"
        thickness = 5
        count_ok += 1
        
    else: # status == 2
        # ĐẬU SAI -> MÀU ĐỎ
        color = (0, 0, 255)
        text = f"O {i+1}: SAI !!!"
        thickness = 5
        count_bad += 1
        
    # Vẽ ô
    cv2.polylines(frame, [slot], True, color, thickness)
    
    # Viết chữ (Canh chỉnh vị trí chữ to rõ)
    text_pos = (slot[0][0], slot[0][1] - 20)
    cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

# --- 4. THU NHỎ & HIỆN ẢNH ---
print(f"\n>>> KẾT QUẢ: {count_ok} xe đậu đúng - {count_bad} xe đậu sai.")
frame_display = cv2.resize(frame, None, fx=0.5, fy=0.5)
cv2.imshow("KET QUA CHECK LOGIC", frame_display)
cv2.waitKey(0)
cv2.destroyAllWindows()