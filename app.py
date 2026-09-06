import streamlit as st
import tempfile
import cv2
import numpy as np
import torch
from ultralytics import YOLO


st.set_page_config(
    page_title="Driving Perception System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(180deg, #0a0e17 0%, #10151f 100%);
    }
    h1 {
        color: #00d9ff;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        border-bottom: 2px solid #00d9ff;
        padding-bottom: 15px;
    }
    .subtitle {
        color: #e8ecf1;
        font-size: 16px;
        margin-bottom: 30px;
    }
    h3 {
        color: #ffffff !important;
    }
    p, li, span, label {
        color: #f0f2f5 !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #f0f2f5 !important;
    }
    .stButton>button {
        background-color: #00d9ff;
        color: #0a0e17;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00b8d9;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.5);
    }
    .metric-card {
        background-color: #161b26;
        border: 1px solid #2a3441;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        color: #00d9ff;
        font-size: 32px;
        font-weight: bold;
    }
    .metric-label {
        color: #e8ecf1;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .video-label {
        color: #ffffff;
        font-size: 15px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    [data-testid="stFileUploader"] {
        background-color: #161b26;
        border: 2px dashed #2a3441;
        border-radius: 10px;
        padding: 20px;
    }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] span {
        color: #f0f2f5 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚗 Real-Time Driving Perception System</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multi-model computer vision pipeline — object detection · multi-object tracking · '
    'monocular depth estimation · lane detection · collision risk analysis</p>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_models():
    model = YOLO("yolov8n.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    midas.to(device)
    return model, midas, transform, device

model, midas, transform, device = load_models()

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

def process_video(input_path, output_path, progress_bar):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    from collections import deque

    depth_history = {}
    smoothed_depths = {}
    warning_counters = {}
    HISTORY_LENGTH = 10
    SMOOTHING_ALPHA = 0.3
    COLLISION_DEPTH_THRESHOLD = 150
    COLLISION_CLOSING_RATE = 15
    WARNING_CONFIRM_FRAMES = 3
    frame_count = 0
    total_detections = 0
    total_warnings_triggered = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        annotated_frame = results[0].plot()
        annotated_frame = detect_lanes(annotated_frame)

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

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            class_name = model.names[int(box.cls[0])]

            box_region = depth_map[y1:y2, x1:x2]
            depth_value = np.mean(box_region) if box_region.size > 0 else 0

            if track_id in smoothed_depths:
                smoothed_depths[track_id] = (
                    SMOOTHING_ALPHA * depth_value + (1 - SMOOTHING_ALPHA) * smoothed_depths[track_id]
                )
            else:
                smoothed_depths[track_id] = depth_value
            current_smoothed = smoothed_depths[track_id]

            if track_id not in depth_history:
                depth_history[track_id] = deque(maxlen=HISTORY_LENGTH)

            raw_danger = False
            if len(depth_history[track_id]) == HISTORY_LENGTH:
                baseline = depth_history[track_id][0]
                depth_change = current_smoothed - baseline
                if current_smoothed > COLLISION_DEPTH_THRESHOLD and depth_change > COLLISION_CLOSING_RATE:
                    raw_danger = True

            depth_history[track_id].append(current_smoothed)

            if raw_danger:
                warning_counters[track_id] = warning_counters.get(track_id, 0) + 1
            else:
                warning_counters[track_id] = max(0, warning_counters.get(track_id, 0) - 1)

            is_warning = warning_counters[track_id] >= WARNING_CONFIRM_FRAMES

            # Now that is_warning exists, it's safe to use it here
            total_detections += 1
            if is_warning:
                total_warnings_triggered.add(track_id)

            color = (0, 0, 255) if is_warning else (0, 255, 0)
            label = f"{class_name} ID:{track_id} D:{current_smoothed:.1f}"
            if is_warning:
                label = "WARNING " + label

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(annotated_frame)

    cap.release()
    out.release()
    return total_detections, total_warnings_triggered

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Upload footage")
    uploaded_file = st.file_uploader("Drop a driving video here", type=["mp4", "avi", "mov"])

with col2:
    st.markdown("### ⚙️ Pipeline stack")
    st.markdown("""
    - **YOLOv8** — object detection
    - **ByteTrack** — multi-object tracking
    - **MiDaS** — monocular depth estimation
    - **Canny + Hough** — lane detection
    - **Custom logic** — collision risk scoring
    """)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    if st.button("▶ Run Perception Pipeline"):
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Processing video...")
        total_detections, warned_ids = process_video(input_path, output_path, progress_bar)
        status_text.empty()

        st.markdown("### 📊 Results")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_detections}</div>'
                        f'<div class="metric-label">Total detections</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(warned_ids)}</div>'
                        f'<div class="metric-label">Collision warnings</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">✓</div>'
                        f'<div class="metric-label">Processing complete</div></div>', unsafe_allow_html=True)

        st.markdown("### 🎬 Before / after")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<p class="video-label">Original upload</p>', unsafe_allow_html=True)
            st.video(input_path)
        with v2:
            st.markdown('<p class="video-label">Processed output</p>', unsafe_allow_html=True)
            st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("⬇ Download processed video", f, file_name="processed_output.mp4")