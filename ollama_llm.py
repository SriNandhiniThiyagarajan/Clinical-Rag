import requests
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "llama3.2"

def ollama_chat_json(system: str, user: str) -> dict:
    """
    Asks Ollama to return a strict JSON object.
    Returns Python dict.
    """
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Ollama supports a "format" parameter for JSON mode
        "format": "json",
    }

    session = requests.Session()
    session.trust_env = False

    r = session.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()

    data = r.json()  # response dict
    content = data["message"]["content"]  # this should be JSON text

    # Convert JSON text -> Python dict
    return json.loads(content)