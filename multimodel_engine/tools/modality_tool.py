# tools/modality_tool.py

from modality_classifier import classify_modality_from_text

def detect_scan_type_from_text(message):
    """
    Wrapper used by the agent.
    """

    return classify_modality_from_text(message)
