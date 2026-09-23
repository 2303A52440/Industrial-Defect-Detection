"""
Unit tests for inspection logging system.
"""

import pytest
import os
import numpy as np
from pathlib import Path
from src.vision.vision_engine import DetectionResult, DefectResult
from src.advisory.executorch_advisor import OperatorAdvisory
from src.logging.inspection_logger import InspectionLogger

def test_logger_sqlite_and_json(tmp_path):
    db_file = tmp_path / "test_inspections.db"
    json_file = tmp_path / "test_inspections.json"
    
    logger = InspectionLogger(db_path=db_file, json_path=json_file)
    
    detection = DetectionResult(
        defects=[DefectResult("scratches", 0.82, (10, 10, 50, 50), "MEDIUM", 1600)],
        annotated_image=np.zeros((100, 100, 3), dtype=np.uint8),
        inference_time_ms=12.5,
        backend_used="OPENCV_ANOMALY_ENGINE",
        pass_quality_check=False
    )
    advisory = OperatorAdvisory(
        action_type="REWORK_SURFACE",
        primary_recommendation="Rework surface abrasions",
        technical_details="Buff surface at Station 4",
        safety_warning=None,
        llama_inference_time_ms=25.0,
        runtime_engine="RULES_ENGINE_FALLBACK",
        tokens_per_second=100.0
    )
    
    rec = logger.log_inspection("CMP-TEST-001", detection, advisory)
    assert rec["qc_status"] == "REJECTED"
    assert rec["primary_defect"] == "scratches"
    
    df = logger.fetch_history_df()
    assert len(df) == 1
    assert df.iloc[0]["component_id"] == "CMP-TEST-001"
