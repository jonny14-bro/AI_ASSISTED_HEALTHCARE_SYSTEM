# agent/memory.py

class PatientMemory:
    def __init__(self):
        self.sessions = {}

    def get_history(self, patient_id):
        return self.sessions.get(patient_id, [])

    def add_message(self, patient_id, role, text):
        if patient_id not in self.sessions:
            self.sessions[patient_id] = []

        self.sessions[patient_id].append({
            "role": role,
            "text": text
        })
