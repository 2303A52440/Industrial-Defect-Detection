"""
Streamlit Factory Floor Operator Dashboard.
Displays real-time visual inspection feeds, Llama 3.2 ExecuTorch advisory alerts,
and system telemetry metrics.
"""

import sys
from pathlib import Path
import time
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import pandas as pd

# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import EdgeQCPipeline

st.set_page_config(
    page_title="Edge-AI Industrial QC Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .metric-card { background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .advisory-reject { background: #7f1d1d; border-left: 5px solid #ef4444; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .advisory-rework { background: #78350f; border-left: 5px solid #f59e0b; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .advisory-pass { background: #064e3b; border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return EdgeQCPipeline()

pipeline = get_pipeline()

# Title Header
st.title("🏭 Edge-AI Industrial Quality Control Assistant")
st.caption("Powered by Raspberry Pi 5 | Hailo NPU Vision Inference | Llama 3.2 ExecuTorch Arm CPU Advisory")

# Sidebar Controls
st.sidebar.header("🕹️ Inspection Controls")

input_source = st.sidebar.radio(
    "Select Component Input Source:",
    [
        "📷 Live Camera (Front / Back / USB)",
        "📁 Upload Image File",
        "🧪 Synthetic Component Generator",
        "🔄 Production Line Loop"
    ]
)

vision_backend_choice = st.sidebar.selectbox(
    "Vision Accelerator Backend:",
    ["Hailo NPU (HailoRT)", "Arm CPU (ONNX Runtime)", "OpenCV Anomaly Detector Engine"]
)

st.sidebar.markdown("---")

def generate_synthetic_metal_surface(defect_type: str = "random") -> np.ndarray:
    """Generates synthetic metal surface texture with controlled defect patterns."""
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    noise = np.random.normal(0, 12, (640, 640)).astype(np.uint8)
    img = cv2.add(img, cv2.merge([noise, noise, noise]))

    if defect_type == "random":
        defect_type = np.random.choice(["clean", "scratches", "crazing", "patches", "pitted_surface", "inclusion"])

    if defect_type == "scratches":
        cv2.line(img, (120, 200), (450, 480), (30, 30, 30), 4)
        cv2.line(img, (140, 190), (470, 470), (40, 40, 40), 2)
    elif defect_type == "crazing":
        for _ in range(8):
            pt1 = (np.random.randint(200, 400), np.random.randint(200, 400))
            pt2 = (pt1[0] + np.random.randint(-40, 40), pt1[1] + np.random.randint(-40, 40))
            cv2.line(img, pt1, pt2, (20, 20, 20), 2)
    elif defect_type == "patches":
        cv2.ellipse(img, (320, 300), (100, 60), 30, 0, 360, (70, 70, 70), -1)
    elif defect_type == "pitted_surface":
        for _ in range(15):
            center = (np.random.randint(150, 500), np.random.randint(150, 500))
            cv2.circle(img, center, np.random.randint(5, 12), (10, 10, 10), -1)
    elif defect_type == "inclusion":
        pts = np.array([[250, 250], [320, 230], [350, 310], [280, 340]], np.int32)
        cv2.fillPoly(img, [pts], (15, 15, 15))

    return img, defect_type

def capture_hardware_camera(device_idx: int) -> np.ndarray:
    """Safely opens hardware camera device index, reads frame, and releases lock."""
    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame
    return None

# Handle Image Acquisition
input_img = None

if input_source == "📷 Live Camera (Front / Back / USB)":
    st.sidebar.subheader("📷 Camera Selection")
    
    camera_type = st.sidebar.radio(
        "Choose Camera:",
        [
            "📱 Browser Camera (Front / Back Flip)",
            "🤳 Front / Built-in Camera (Device 0)",
            "📷 Back / USB Inspection Camera (Device 1)",
            "⚙️ Custom Camera Index"
        ]
    )

    if camera_type == "📱 Browser Camera (Front / Back Flip)":
        st.sidebar.caption("Use your browser photo tool to snap a frame or switch between Front/Rear camera.")
        cam_photo = st.sidebar.camera_input("Take Photo from Camera", key="browser_camera")
        if cam_photo is not None:
            pil_image = Image.open(cam_photo).convert("RGB")
            input_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            st.info("📷 Click **'Take Photo'** in the sidebar to run inspection on a live camera frame.")
            input_img, _ = generate_synthetic_metal_surface("clean")

    elif camera_type == "🤳 Front / Built-in Camera (Device 0)":
        st.sidebar.caption("Accessing primary Front / Built-in Camera (Index 0)")
        if st.sidebar.button("📸 Capture Front Camera Frame", use_container_width=True):
            frame = capture_hardware_camera(0)
            if frame is not None:
                input_img = frame
                st.sidebar.success("Captured frame from Front Camera (Device 0)!")
            else:
                st.sidebar.error("Front Camera (Device 0) not accessible. Please check webcam connection.")
                input_img, _ = generate_synthetic_metal_surface("clean")
        else:
            frame = capture_hardware_camera(0)
            if frame is not None:
                input_img = frame
            else:
                input_img, _ = generate_synthetic_metal_surface("clean")

    elif camera_type == "📷 Back / USB Inspection Camera (Device 1)":
        st.sidebar.caption("Accessing secondary Back / USB Inspection Camera (Index 1)")
        if st.sidebar.button("📸 Capture Back Camera Frame", use_container_width=True):
            frame = capture_hardware_camera(1)
            if frame is not None:
                input_img = frame
                st.sidebar.success("Captured frame from Back Camera (Device 1)!")
            else:
                st.sidebar.warning("Back Camera (Device 1) not found. Connecting to default camera (Device 0)...")
                frame_0 = capture_hardware_camera(0)
                input_img = frame_0 if frame_0 is not None else generate_synthetic_metal_surface("scratches")[0]
        else:
            frame = capture_hardware_camera(1)
            if frame is not None:
                input_img = frame
            else:
                # Fallback to Device 0 if device 1 not present
                frame_0 = capture_hardware_camera(0)
                input_img = frame_0 if frame_0 is not None else generate_synthetic_metal_surface("scratches")[0]

    else: # Custom Camera Index
        custom_idx = st.sidebar.number_input("Hardware Camera Device Index:", min_value=0, max_value=10, value=0, step=1)
        if st.sidebar.button(f"📸 Capture Camera Device {custom_idx}", use_container_width=True):
            frame = capture_hardware_camera(int(custom_idx))
            if frame is not None:
                input_img = frame
                st.sidebar.success(f"Captured frame from Camera {custom_idx}!")
            else:
                st.sidebar.error(f"Could not open Camera Device {custom_idx}.")
                input_img, _ = generate_synthetic_metal_surface("scratches")
        else:
            frame = capture_hardware_camera(int(custom_idx))
            input_img = frame if frame is not None else generate_synthetic_metal_surface("clean")[0]

elif input_source == "📁 Upload Image File":
    st.sidebar.subheader("📁 Image File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Select surface image file...",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"]
    )
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            input_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            st.sidebar.success(f"File loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error reading image: {e}")
            input_img, _ = generate_synthetic_metal_surface("scratches")
    else:
        st.info("📁 Upload an image file in the sidebar to analyze a component.")
        input_img, _ = generate_synthetic_metal_surface("scratches")

elif input_source == "🧪 Synthetic Component Generator":
    st.sidebar.subheader("🧪 Synthetic Test Patterns")
    defect_seed = st.sidebar.selectbox(
        "Select Defect Pattern:",
        ["scratches", "crazing", "patches", "pitted_surface", "inclusion", "clean", "random"]
    )
    input_img, actual_type = generate_synthetic_metal_surface(defect_seed)

else:
    # Simulated Production Loop
    input_img, _ = generate_synthetic_metal_surface("random")
    time.sleep(0.5)
    st.rerun()

# Main Interface Layout
col_left, col_right = st.columns([3, 2])

if input_img is not None:
    # Execute Pipeline
    detection, advisory, log_record = pipeline.process_component_inspection(input_img)

    with col_left:
        st.subheader("📹 Component Visual Inspection Feed")
        
        # Display Annotated Frame
        rgb_annotated = cv2.cvtColor(detection.annotated_image, cv2.COLOR_BGR2RGB)
        st.image(
            rgb_annotated,
            use_column_width=True,
            caption=f"Component ID: {log_record['component_id']} | QC Status: {log_record['qc_status']}"
        )
        
        # Telemetry metrics under visual feed
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("QC Decision", log_record["qc_status"], delta="Pass" if log_record["qc_status"]=="PASSED" else "-Reject")
        m2.metric("Vision Latency", f"{detection.inference_time_ms:.1f} ms", f"{1000/max(detection.inference_time_ms,1):.0f} FPS")
        m3.metric("Defects Found", f"{len(detection.defects)}")
        m4.metric("Vision Backend", detection.backend_used.split("_")[0])

    with col_right:
        st.subheader("🤖 On-Device Llama 3.2 ExecuTorch Advisory")
        
        # Advisory Card Styling
        card_class = "advisory-reject" if advisory.action_type == "REJECT_COMPONENT" else "advisory-rework" if advisory.action_type == "REWORK_SURFACE" else "advisory-pass"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin-top:0; color: #ffffff;">{advisory.action_type.replace('_', ' ')}</h3>
            <p style="font-size: 1.1em; font-weight: bold; color: #ffffff;">{advisory.primary_recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Operator Technical Action Plan")
        st.info(advisory.technical_details)

        if advisory.safety_warning:
            st.warning(advisory.safety_warning)

        st.markdown("---")
        st.markdown("### ⚡ ExecuTorch Engine Telemetry")
        t1, t2 = st.columns(2)
        t1.metric("Llama Inference Time", f"{advisory.llama_inference_time_ms:.1f} ms")
        t2.metric("Generation Speed", f"{advisory.tokens_per_second:.1f} tok/s")
        st.caption(f"Runtime Engine: `{advisory.runtime_engine}`")

# Historic Inspection Table
st.markdown("---")
st.subheader("📊 Factory Inspection Logs & Analytics")

history_df = pipeline.logger.fetch_history_df(limit=25)
if not history_df.empty:
    st.dataframe(
        history_df,
        column_config={
            "timestamp": "Timestamp",
            "component_id": "Part ID",
            "qc_status": "QC Status",
            "defect_count": "Defects",
            "primary_defect": "Primary Defect",
            "max_severity": "Severity",
            "vision_latency_ms": "Vision (ms)",
            "llama_latency_ms": "Llama (ms)",
            "recommendation": "Operator Instruction"
        },
        use_container_width=True,
        hide_index=True
    )
    
    csv_data = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Inspection History (CSV)", csv_data, "inspection_log_export.csv", "text/csv")
else:
    st.write("No inspection records logged yet.")
