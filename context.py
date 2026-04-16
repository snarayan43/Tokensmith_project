import json
import os
from datetime import datetime

class UserContextStore:
    def __init__(self, storage_path="context_store.json", persona_path="user_persona.txt"):
        self.storage_path = storage_path
        self.persona_path = persona_path
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return []

    def save_query(self, query, answer):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer
        }
        self.history.append(entry)
        
        with open(self.storage_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        # Trigger summary every 5 queries
        if len(self.history) % 5 == 0:
            return True 
        return False

    def get_persona(self):
        if os.path.exists(self.persona_path):
            with open(self.persona_path, 'r') as f:
                return f.read()
        return "New user with no established history."

    def update_persona_file(self, new_summary):
        with open(self.persona_path, 'w') as f:
            f.write(new_summary)