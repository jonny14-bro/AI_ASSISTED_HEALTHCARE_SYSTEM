# agent/medical_agent.py

from agent.memory import PatientMemory
from agent.conversation import generate_reply
from agent.triage import assess_risk

from tools.modality_tool import detect_scan_type_from_text
from tools.xray_tool import analyze_xray
from tools.ultrasound_tool import analyze_ultrasound
from tools.validator import validate_input

from reports.report_generator import save_report
from reports.report_formatter import format_clinical_report


class MedicalAIAgent:

    def __init__(self):
        self.memory = PatientMemory()

    def handle_request(self, patient_id, message, image_path=None):

        # -------- VALIDATION --------
        validate_input(patient_id, message, image_path)

        # -------- STORE PATIENT MESSAGE --------
        self.memory.add_message(patient_id, "patient", message)

        # -------- GET HISTORY --------
        history = self.memory.get_history(patient_id)

        # ==========================================================
        # TEXT-ONLY CONVERSATION
        # ==========================================================
        if not image_path:
            reply = generate_reply(message, history)

            self.memory.add_message(patient_id, "agent", reply)

            return {
                "reply": reply,
                "report_path": None,
                "clinical_report": None
            }

        # ==========================================================
        # DETECT MODALITY FROM TEXT
        # ==========================================================
        modality = detect_scan_type_from_text(message)

        if modality is None:
            reply = (
                "Please specify whether this is an X-ray "
                "or ultrasound scan."
            )

            self.memory.add_message(patient_id, "agent", reply)

            return {
                "reply": reply,
                "report_path": None,
                "clinical_report": None
            }

        # ==========================================================
        # RUN APPROPRIATE TOOL
        # ==========================================================
        if modality == "xray":
            findings, confidence, output_dir, report_id = analyze_xray(
                image_path, patient_id
            )
        else:
            findings, confidence, output_dir, report_id = analyze_ultrasound(
                image_path, patient_id
            )

        # ==========================================================
        # RISK ASSESSMENT
        # ==========================================================
        risk = assess_risk(message, findings)

        # ==========================================================
        # SAVE STATIC REPORT (for backend / dataset)
        # ==========================================================
        report_path = save_report(
            report_id=report_id,
            patient_id=patient_id,
            modality=modality,
            findings=findings,
            confidence=confidence,
            output_image=output_dir
        )

        # ==========================================================
        # FORMAT DOCTOR-READABLE REPORT
        # ==========================================================
        clinical_report = format_clinical_report(report_path)

        # ==========================================================
        # GENERATE HUMAN-LIKE RESPONSE
        # ==========================================================
        reply = generate_reply(
            message,
            history,
            modality,
            findings,
            confidence,
            risk
        )

        # -------- STORE AGENT REPLY --------
        self.memory.add_message(patient_id, "agent", reply)

        # ==========================================================
        # RETURN STRUCTURED OUTPUT
        # ==========================================================
        return {
            "reply": reply,
            "report_path": report_path,
            "clinical_report": clinical_report
        }
