import cv2
from ultralytics import YOLO

# Load YOLO model (lần đầu sẽ tự tải)
model = YOLO("yolov8n.pt")

# Mở webcam laptop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được webcam")
    exit()

print("Đang chạy YOLO... Nhấn Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect xe: car, motorbike, bus, truck
    results = model(frame, classes=[2, 3, 5, 7], conf=0.4)

    # Vẽ bounding box
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Parking AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
