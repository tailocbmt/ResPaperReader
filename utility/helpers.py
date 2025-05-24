import json
import logging
from typing import Dict


def load_prompts(file_path="prompts/system_prompts.json") -> Dict[str, Dict[str, str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load system prompts: {e}")
        return {}
