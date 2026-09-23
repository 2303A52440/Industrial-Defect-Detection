"""
Unit tests for ExecuTorch Llama operator advisory engine.
"""

import pytest
import numpy as np
from src.vision.vision_engine import VisionEngine, DetectionResult, DefectResult
from src.advisory.executorch_advisor import ExecuTorchLlamaAdvisor, OperatorAdvisory

def test_advisor_clean_detection():
    advisor = ExecuTorchLlamaAdvisor()
    detection = DetectionResult(
        defects=[],
        annotated_image=np.zeros((640, 640, 3), dtype=np.uint8),
        inference_time_ms=10.0,
        backend_used="OPENCV_ANOMALY_ENGINE",
        pass_quality_check=True
    )
    
    advisory: OperatorAdvisory = advisor.generate_advisory(detection)
    assert advisory.action_type == "PASS_COMPONENT"
    assert "passed" in advisory.primary_recommendation.lower()

def test_advisor_critical_defect():
    advisor = ExecuTorchLlamaAdvisor()
    critical_defect = DefectResult(
        class_name="crazing",
        confidence=0.88,
        bbox=(100, 100, 300, 300),
        severity="CRITICAL",
        area_pixels=40000
    )
    detection = DetectionResult(
        defects=[critical_defect],
        annotated_image=np.zeros((640, 640, 3), dtype=np.uint8),
        inference_time_ms=15.0,
        backend_used="OPENCV_ANOMALY_ENGINE",
        pass_quality_check=False
    )
    
    advisory: OperatorAdvisory = advisor.generate_advisory(detection)
    assert advisory.action_type == "REJECT_COMPONENT"
    assert "CRITICAL" in advisory.primary_recommendation
