# agent/conversation.py

def generate_reply(message, history=None,
                   modality=None,
                   findings=None,
                   confidence=None,
                   risk="LOW"):

    msg = message.lower()

    empathetic = any(w in msg for w in
                     ["pain", "hurt", "injury", "worried", "scared"])

    # ---------- IMAGE ANALYSIS RESPONSE ----------
    if modality and findings:

        modality_name = "X-ray" if modality == "xray" else "Ultrasound"

        intro = ""

        if empathetic:
            intro = (
                "I’m really sorry you’re going through this. "
                "I’ve carefully reviewed your scan. "
            )
        else:
            intro = "I’ve analyzed the image you shared. "

        risk_msg = ""
        if risk == "HIGH":
            risk_msg = (
                "\n⚠️ Based on your symptoms, this could require "
                "urgent medical attention."
            )

        explanation = (
            f"{intro}It appears to be a {modality_name} scan.\n\n"
            f"Here’s what I found:\n"
            f"{findings}\n\n"
            f"Model confidence is about {confidence:.1f}%."
            f"{risk_msg}\n\n"
            "If you’re experiencing significant discomfort or "
            "symptoms are worsening, please seek professional "
            "medical care as soon as possible."
        )

        followup = (
            "\n\nIf you want, you can tell me your symptoms or "
            "when the injury happened, and I’ll try to guide you."
        )

        return explanation + followup

    # ---------- TEXT-ONLY RESPONSE ----------

    if empathetic:
        return (
            "I’m sorry you’re feeling this way. "
            "If you have a scan image (X-ray or ultrasound), "
            "you can upload it and I’ll help analyze it. "
            "You can also tell me more about what happened."
        )

    return (
        "I’m here to help. If you upload a medical scan "
        "(for example an X-ray or ultrasound), I can analyze it. "
        "You can also describe your symptoms."
    )
