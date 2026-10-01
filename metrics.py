import json
import os
import threading
from datetime import datetime, timezone


METRICS_FILE = os.getenv("METRICS_FILE", "logs/metrics.jsonl")

_lock = threading.Lock()


def timestamp_now():
    """
    Devuelve la marca de tiempo actual en ISO 8601 (UTC).
    """

    return datetime.now(timezone.utc).isoformat()


def save_record(record):
    """
    Anexa un registro de turno como una línea JSON al archivo de métricas.

    El registro se conserva tal cual se recibe para poder calcular
    métricas offline posteriormente (word count, formalidad, etc.).
    """

    os.makedirs(os.path.dirname(METRICS_FILE) or ".", exist_ok=True)

    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

    with _lock:
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(line)