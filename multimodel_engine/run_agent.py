# run_agent.py

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from agent.medical_agent import MedicalAIAgent


def main():

    agent = MedicalAIAgent()

    print("=== Medical AI Assistant ===")

    patient_id = input("Enter Patient ID: ").strip()

    while True:

        message = input("\nYou: ").strip()

        if message.lower() in ["exit", "quit"]:
            print("Session ended.")
            break

        if message == "":
            continue

        image_path = input(
            "Enter image path (or press Enter for none): "
        ).strip()

        if image_path == "":
            image_path = None

        result = agent.handle_request(
            patient_id=patient_id,
            message=message,
            image_path=image_path
        )

        print("\nAssistant:")
        print(result["reply"])

        if result["report_path"]:
            print("\nReport saved at:", result["report_path"])


if __name__ == "__main__":
    main()
