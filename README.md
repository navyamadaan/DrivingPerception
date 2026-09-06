# 🚗 Real-Time Driving Perception System

A multi-model computer vision pipeline that mirrors the core perception layer of modern ADAS (Advanced Driver-Assistance Systems) — built end-to-end using open-source models and a single camera feed, no LiDAR or proprietary sensors required.

**[Live demo →](https://your-app-name.streamlit.app)** *(replace with your actual Streamlit Cloud URL)*

---

## What it does

Upload a driving video and the pipeline will:

- **Detect** vehicles, pedestrians, and other road objects (YOLOv8)
- **Track** each object with a persistent ID across frames (ByteTrack)
- **Estimate depth** for every object using a single camera (MiDaS)
- **Detect lane markings** on the road (Canny edge detection + Hough Transform)
- **Flag collision risk** — objects that are both close *and* closing in fast get highlighted in red with a warning label

The processed video, with all annotations overlaid, is shown side-by-side with the original and available to download.

---

## Tech stack

| Component | Purpose |
|---|---|
| **YOLOv8** (Ultralytics) | Real-time object detection |
| **ByteTrack** | Multi-object tracking with persistent IDs |
| **MiDaS (small)** | Monocular depth estimation |
| **OpenCV (Canny + Hough Transform)** | Classical lane line detection |
| **PyTorch** | Model inference backend |
| **Streamlit** | Web app / interactive deployment |

---

## How the collision warning works

Rather than flagging any nearby object, the system looks for objects that are **both close and actively approaching** — closer to how a real collision-warning system should behave:

1. Depth is sampled across each object's full bounding box (not a single pixel) and smoothed over time with an exponential moving average, since raw MiDaS depth is noisy frame-to-frame.
2. Each object's current depth is compared against its depth from ~10 frames ago (a rolling baseline) rather than the immediately previous frame — this filters out natural frame-to-frame oscillation and captures the real approach trend.
3. A warning only fires after several consecutive frames confirm the object is both past a proximity threshold and closing in — a single noisy frame can't trigger a false alarm, and a single clean frame can't instantly cancel a real one (the counter decays gradually instead of resetting).

## Architecture

```
Video input
     │
     ├──► YOLOv8 detection ──► ByteTrack (assigns persistent IDs)
     │
     ├──► MiDaS depth estimation ──► per-object depth (box-averaged, smoothed)
     │
     ├──► Canny edge detection ──► ROI mask ──► Hough Transform ──► lane lines
     │
     └──► Fusion layer
              │
              ├─ Label each tracked object: class, ID, depth
              ├─ Compare depth trend over rolling window
              └─ Flag + highlight collision risk (red) vs normal (green)
                       │
                       ▼
              Annotated output video
```

---

## Running locally

```bash
git clone https://github.com/navyamadaan/DrivingPerception.git
cd DrivingPerception
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`), upload a driving video, and click **Run Perception Pipeline**.

> First run downloads YOLOv8 and MiDaS weights automatically — this takes a minute.

---

## Project structure

```
DrivingPerception/
├── app.py              # Streamlit web app (upload → process → view/download)
├── src/
│   └── main.py          # Standalone local script (webcam or video file, live windows)
├── requirements.txt
└── README.md
```

---

## Known limitations

- MiDaS depth is **relative**, not calibrated to real-world meters — the collision warning reacts to relative closing trends, not exact distances.
- Runs on CPU in the deployed version (Streamlit Cloud has no GPU), so processing is not real-time for longer clips — expect roughly a minute or two of processing per short video.
- This is a proof-of-concept for the perception layer behind ADAS features, not a safety-certified or production-ready system.

---

## What I'd add next

- Speed/closing-rate estimation displayed numerically, not just as a binary warning
- Lane departure detection (is the vehicle drifting out of its lane, not just where the lanes are)
- Event logging to CSV for post-drive analytics (detections over time, near-miss counts)
- A calibration step to convert MiDaS's relative depth into approximate real-world distance

---

Built by [Navya Madaan](https://github.com/navyamadaan)
