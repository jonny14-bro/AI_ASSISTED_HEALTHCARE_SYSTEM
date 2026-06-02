# reports/report_formatter.py

import json


def format_clinical_report(report_path):

    with open(report_path) as f:
        r = json.load(f)

    modality = r["modality"].upper()

    return f"""
=== CLINICAL AI REPORT ===

Patient ID: {r['patient_id']}
Report ID: {r['report_id']}
Modality: {modality}

Findings:
{r['findings']}

Model Confidence: {r['confidence']}%

---------------------------------
For clinical decision support only
---------------------------------
"""
