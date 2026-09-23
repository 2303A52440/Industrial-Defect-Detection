# 🏭 Edge-AI Industrial Quality Control Assistant

An end-to-end, on-device AI system for real-time visual inspection of manufactured components, defect detection, and intelligent operator assistance powered by **Llama 3.2** on **ExecuTorch** and **Hailo NPU / OpenCV** vision pipelines on Raspberry Pi 5 / Arm SoCs.

---

## 🖥️ Interactive Prototype Showcase

The prototype includes a production-grade Web Dashboard for factory floor operators built with Streamlit:

- **🤳 Front Camera Stream**: Direct capture from primary webcam / user-facing camera (Device 0).
- **📷 Back / USB Inspection Camera Stream**: Direct capture from secondary rear camera / external USB inspection endoscope (Device 1).
- **📱 Browser Camera Tool**: One-click camera input with automatic Front/Back flip capability.
- **📁 Universal File Upload**: Upload any `.jpg`, `.png`, `.bmp`, `.webp` surface image for real-time defect analysis.
- **🤖 Llama ExecuTorch Advisory Cards**: Instant operator remediation instructions (`REJECT_COMPONENT`, `REWORK_SURFACE`, `PASS_WITH_MONITORING`).
- **⚡ System Telemetry**: Live FPS, Vision Latency (ms), Llama Token Speed (tok/s), and SQLite database CSV logging.

---

## 🌟 Key Features

- **⚡ Dual-Pipeline Edge Architecture**:
  - **Vision Engine**: Real-time defect detection (cracks, scratches, crazing, pitted surfaces, inclusions) running via HailoRT on Hailo NPU or CPU ONNX/OpenCV fallback.
  - **Llama ExecuTorch Advisory**: On-device Llama 3.2 model compiled for ExecuTorch running on Arm CPU to provide actionable, safety-compliant operator recommendations.
- **🖥️ Factory Floor Operator Dashboard**: Streamlit interface displaying real-time annotated camera feeds, Llama advisory cards, and telemetry (FPS, Latency, Token Speed).
- **📊 Inspection Telemetry & Logging**: Automatic logging of component IDs, defect severity, vision metrics, and advisory actions into SQLite and JSON.
- **🧪 CLI Demo Simulator & Dataset Generators**: Out-of-the-box synthetic component stream simulator and sample dataset tools.

---

## 📁 Project Structure

```
edge_industrial_qc/
├── src/
│   ├── config.py                 # System configuration & thresholds
│   ├── pipeline.py               # Dual-pipeline coordinator
│   ├── vision/
│   │   └── vision_engine.py      # Hailo NPU / ONNX / OpenCV defect detector
│   ├── advisory/
│   │   └── executorch_advisor.py # Llama 3.2 ExecuTorch advisory module
│   ├── logging/
│   │   └── inspection_logger.py  # SQLite DB & JSON inspection logger
│   └── dashboard/
│       └── app.py                # Streamlit Web UI for operators
├── scripts/
│   ├── demo_simulator.py         # Batch CLI inspection demo
│   └── download_datasets.py      # Sample surface defect dataset builder
├── tests/                        # Pytest test suite
│   ├── test_vision.py
│   ├── test_advisor.py
│   └── test_logger.py
├── docs/
│   └── TECHNICAL_REPORT.md       # Full project technical report
├── models/                       # Models directory (ONNX / HEF / PTE)
├── data/                         # Datasets & sample images
├── logs/                         # SQLite database & JSON logs
└── requirements.txt
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone/copy the workspace and install python dependencies:

```bash
cd edge_industrial_qc
pip install -r requirements.txt
```

### 2. Generate Sample Surface Dataset
```bash
python scripts/download_datasets.py
```

### 3. Run Automated Tests
```bash
pytest
```

### 4. Run CLI Inspection Demo
```bash
python scripts/demo_simulator.py
```

### 5. Launch Operator Dashboard
```bash
python -m streamlit run src/dashboard/app.py
```

---

## 📄 License & Technical Specifications

- **Target Hardware**: Raspberry Pi 5 (Arm Cortex-A76), Arm-Linux / Android devices.
- **Accelerator**: Hailo AI HAT+ (Hailo-8L / Hailo-8 NPU) via HailoRT.
- **LLM Runtime**: ExecuTorch (PyTorch Edge) with Llama 3.2 1B/3B Instruct.
- **Default Dataset**: NEU Surface Defect / MVTec AD standard.
