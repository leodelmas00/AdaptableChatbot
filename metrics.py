import json
import os
import re
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
    Anexa un registro como una línea JSON al archivo de métricas.

    El registro se conserva tal cual se recibe para poder calcular
    métricas offline posteriormente (word count, formalidad, etc.).
    """

    os.makedirs(os.path.dirname(METRICS_FILE) or ".", exist_ok=True)

    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

    with _lock:
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(line)


def log_event(event_type, **fields):
    """
    Registra un evento tipado (turn, session_start, config_change,
    survey_response, session_end, etc.) con su timestamp.
    """

    record = {
        "event_type": event_type,
        "timestamp": timestamp_now(),
    }
    record.update(fields)

    save_record(record)


# ============================================================
# MÉTRICAS LINGÜÍSTICAS (cómputo puro sobre texto)
# ============================================================

def word_count(text):
    """
    Cantidad de palabras de un texto.
    """

    if not text:
        return 0

    return len(str(text).split())


def sentence_count(text):
    """
    Cantidad aproximada de oraciones de un texto.
    """

    if not text:
        return 0

    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())

    return len([s for s in sentences if s.strip()])


HEDGE_TERMS = [
    "maybe", "might", "perhaps", "possibly", "probably", "likely",
    "seems", "seem", "seemed", "appears", "appear", "appeared",
    "could", "would", "may", "somewhat", "sort of", "kind of",
    "rather", "quite", "almost", "about", "approximately", "around",
    "i think", "i believe", "i guess", "i suppose", "in my opinion",
    "to some extent", "somehow", "more or less", "partly", "fairly",
    "generally", "usually", "often", "sometimes", "occasionally",
    "suggest", "suggests", "suggested", "tends to", "tend to",
    "a bit", "a little", "basically", "essentially", "mostly",
    "pretty much", "arguably", "allegedly", "supposedly", "seemingly",
    "reportedly", "hard to say", "not sure", "unsure", "uncertain",
    "as far as i know", "if i recall", "i'd say",
]


def hedge_count(text):
    """
    Cantidad de términos de hedging presentes en un texto.
    """

    if not text:
        return 0

    lowered = str(text).lower()
    count = 0

    for term in HEDGE_TERMS:
        count += len(re.findall(rf"\b{re.escape(term)}\b", lowered))

    return count


def hedge_density(text):
    """
    Densidad de hedging: hedges / cantidad de palabras.
    """

    total = word_count(text)

    if total == 0:
        return 0.0

    return hedge_count(text) / total


# ============================================================
# CUMPLIMIENTO DE LAS INSTRUCCIONES DEL PROMPT
# ============================================================

LENGTH_TARGETS = {
    "Short": (30, 60),
    "Long": (140, 200),
}


def check_length_compliance(words, target):
    """
    Indica si la cantidad de palabras cumple el rango del target.
    Devuelve None si no hay target aplicable.
    """

    if target is None:
        return None

    low, high = LENGTH_TARGETS.get(target, (None, None))

    if low is None or words is None:
        return None

    return low <= words <= high


def check_formality_compliance(label, target):
    """
    Indica si el registro detectado coincide con el target.
    Devuelve None si no hay target aplicable.
    """

    if target is None:
        return None

    if target == "Formal":
        return label == "formal"

    if target == "Informal":
        return label == "informal"

    return None