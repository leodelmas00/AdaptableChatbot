"""
Clasificador de formalidad basado en un transformer.

Modelo: s-nlp/roberta-base-formality-ranker
    (roberta-base, text-classification formal/informal).

El modelo se carga de forma perezosa (lazy) la primera vez que se usa
y se cachea para el resto del proceso.
"""

MODEL_NAME = "s-nlp/roberta-base-formality-ranker"

_tokenizer = None
_model = None


def _load():
    """
    Carga (una sola vez) el tokenizer y el modelo desde HuggingFace.
    """

    global _tokenizer, _model

    if _model is None:
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    return _tokenizer, _model


def classify_formality(text):
    """
    Clasifica la formalidad de un texto.

    Devuelve (label, score) donde label es "formal" o "informal" y
    score es la probabilidad de la clase formal en [0, 1].
    """

    if not text:
        return None, None

    import torch

    tokenizer, model = _load()

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]

    id2label = model.config.id2label or {}

    formal_idx = None
    for idx, label in id2label.items():
        if str(label).lower() == "formal":
            formal_idx = int(idx)
            break

    if formal_idx is None:
        formal_idx = 1

    score = float(probs[formal_idx])
    label = "formal" if score >= 0.5 else "informal"

    return label, score