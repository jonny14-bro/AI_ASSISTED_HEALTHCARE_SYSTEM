# tools/ultrasound_tool.py

import json
from ultra_sound_detection_engine.inference.test_single_scan import ultrasound_scan
import os
from datetime import datetime


def analyze_ultrasound(image_path, patient_id):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scan_dir = os.path.join("patients", patient_id, "ultrasound", timestamp)
    os.makedirs(scan_dir, exist_ok=True)

    ultrasound_scan(image_path, scan_dir, patient_id)

    result_json = os.path.join(scan_dir, "result.json")

    with open(result_json) as f:
        result = json.load(f)

    confidence = result["confidence"]

    if confidence > 80:
        findings = "Suspicious lesion detected."
    elif confidence > 60:
        findings = "Indeterminate lesion detected. Recommend further evaluation."
    elif confidence > 40:
        findings = "Probably benign lesion detected. Consider routine follow-up. Recommend further evaluation."
    else:
        findings = "No abnormality detected."

    return findings, confidence, scan_dir, timestamp
