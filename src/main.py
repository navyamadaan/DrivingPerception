import cv2
from ultralytics import YOLO
import torch
import numpy as np

model = YOLO("yolov8n.pt")

VIDEO_SOURCE = "data/vdo2.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
midas.to(device)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# --- Collision warning state ---
previous_depths = {}
smoothed_depths = {}
warning_counters = {}

SMOOTHING_ALPHA = 0.3
COLLISION_DEPTH_THRESHOLD = 150
COLLISION_CLOSING_RATE = 3
WARNING_CONFIRM_FRAMES = 4


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    roi_vertices = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi / 180,
                              threshold=50, minLineLength=40, maxLineGap=100)

    line_image = np.zeros_like(frame)
    left_lines = []
    right_lines = []

    if lines is not None:
        lines = lines.reshape(-1, 4)
        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    def average_line(line_list):
        if not line_list:
            return None
        x1s, y1s, x2s, y2s = zip(*line_list)
        return int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))

    left_avg = average_line(left_lines)
    right_avg = average_line(right_lines)

    for avg_line in [left_avg, right_avg]:
        if avg_line is not None:
            x1, y1, x2, y2 = avg_line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 8)

    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return combined


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # 1. Detection + tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    annotated_frame = results[0].plot()
    annotated_frame = detect_lanes(annotated_frame)

    # 2. Depth estimation
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # 3. Fuse detection + depth + collision warning
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        class_name = model.names[int(box.cls[0])]

        box_region = depth_map[y1:y2, x1:x2]
        depth_value = np.mean(box_region) if box_region.size > 0 else 0

        # Smooth depth over time to reduce frame-to-frame noise
        if track_id in smoothed_depths:
            smoothed_depths[track_id] = (
                SMOOTHING_ALPHA * depth_value + (1 - SMOOTHING_ALPHA) * smoothed_depths[track_id]
            )
        else:
            smoothed_depths[track_id] = depth_value
        current_smoothed = smoothed_depths[track_id]

        # Check if this object is close AND closing in fast
        raw_danger = False
        if track_id in previous_depths:
            depth_change = current_smoothed - previous_depths[track_id]
            if current_smoothed > COLLISION_DEPTH_THRESHOLD and depth_change > COLLISION_CLOSING_RATE:
                raw_danger = True

        # Require several consecutive dangerous frames before confirming a warning
        if raw_danger:
            warning_counters[track_id] = warning_counters.get(track_id, 0) + 1
        else:
            warning_counters[track_id] = 0

        is_warning = warning_counters[track_id] >= WARNING_CONFIRM_FRAMES

        previous_depths[track_id] = current_smoothed

        # Draw box + label, red if warning, green otherwise
        color = (0, 0, 255) if is_warning else (0, 255, 0)
        label = f"{class_name} ID:{track_id} D:{current_smoothed:.1f}"
        if is_warning:
            label = "WARNING " + label

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. Depth map visualization window
    depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

    # 5. Show windows
    cv2.imshow("Driving Perception - Tracking", annotated_frame)
    cv2.moveWindow("Driving Perception - Tracking", 50, 50)

    cv2.imshow("Depth Map", depth_display)
    cv2.moveWindow("Depth Map", 700, 50)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()