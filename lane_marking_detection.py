import cv2
import numpy as np
from ultralytics import YOLO
import os

# ----------------------------------------
# 1. Download YOLOv8 model automatically
# ----------------------------------------
MODEL_PATH = "yolov8n.pt"
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv8 model...")
    from urllib.request import urlretrieve
    urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt", MODEL_PATH)

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# ----------------------------------------
# Lane Detection Function
# ----------------------------------------
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=50,
                            minLineLength=50, maxLineGap=150)
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

# ----------------------------------------
# Main Loop - Webcam / Video
# ----------------------------------------
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "14246831_3840_2160_30fps.mp4")
cap = cv2.VideoCapture(VIDEO_PATH)  # Using absolute path to video file
if not cap.isOpened():
    raise FileNotFoundError(f"Video file not found or cannot be opened: {VIDEO_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    print("Frame shape:", frame.shape)  # Add this line to check resolution
    # Detect Lanes
    lane_frame = detect_lanes(frame)

    # Detect Objects with YOLO
    results = model.predict(lane_frame, conf=0.4)
    annotated_frame = results[0].plot()

    # Show Output
    # Calculate display size while maintaining aspect ratio
    max_display_width = 1280  # You can adjust this value
    scale_factor = max_display_width / frame.shape[1]
    display_width = int(frame.shape[1] * scale_factor)
    display_height = int(frame.shape[0] * scale_factor)
    
    # Resize frame for display
    display_frame = cv2.resize(annotated_frame, (display_width, display_height))
    
    # Create window with resize property
    cv2.namedWindow("Lane & Object Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Lane & Object Detection", display_frame)
    
    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
