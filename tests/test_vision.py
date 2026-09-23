"""
Unit tests for Vision Engine defect detection.
"""

import pytest
import numpy as np
import cv2
from src.vision.vision_engine import VisionEngine, DetectionResult

def test_vision_engine_initialization():
    engine = VisionEngine(backend="auto")
    assert engine.active_backend in ["HAILO_NPU", "ONNX_ARM_CPU", "OPENCV_ANOMALY_ENGINE"]

def test_vision_engine_clean_frame():
    engine = VisionEngine(backend="auto")
    # Clean grey image
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    res: DetectionResult = engine.inspect_frame(img)
    
    assert res.inference_time_ms >= 0.0
    assert isinstance(res.annotated_image, np.ndarray)
    assert res.pass_quality_check is True

def test_vision_engine_defective_frame():
    engine = VisionEngine(backend="auto")
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    # Add dark scratch
    cv2.line(img, (100, 100), (450, 450), (20, 20, 20), 5)
    
    res: DetectionResult = engine.inspect_frame(img)
    assert len(res.defects) > 0
    assert res.pass_quality_check is False
