# log_manager.py - Simple logging system for RAG app
import os
import json
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "query_logs.jsonl")

def init_logs():
    """Ensure logs folder and file exist."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            pass

def append_log(log_type, data):
    """Append a log entry as JSONL."""
    init_logs()
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": log_type,
        "data": data
    }
    
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"Failed to write log: {e}")
        return False

def load_logs(max_entries=2000):
    """Load logs as a Python list."""
    init_logs()
    logs = []
    
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    
    return logs[-max_entries:]

def clear_logs():
    """Clear all logs."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    init_logs()
    return True