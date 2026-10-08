"""
Instrumentos de encuesta post-interacción (Frustration NASA-TLX,
S-TIAS y CSAT) junto con la construcción de su interfaz Gradio y el
registro de las respuestas.

Los ítems usan las versiones estándar publicadas y están en inglés.
"""

import gradio as gr

from metrics import log_event


# ============================================================
# DEFINICIONES DE INSTRUMENTOS
# ============================================================

# NASA-TLX (Raw TLX): solo la dimensión de Frustración, escala 0-100.
NASA_TLX_ITEMS = [
    {
        "id": "nasatlx_frustration",
        "label": "Frustration",
        "description": (
            "How insecure, discouraged, irritated, stressed, and "
            "annoyed did you feel during the task?"
        ),
    },
]

# S-TIAS (Short Trust in Automation Scale): 3 ítems, Likert 1-7.
STIAS_ITEMS = [
    "I am confident in the AI assistant.",
    "The AI assistant is reliable.",
    "I can trust the AI assistant.",
]

STIAS_CHOICES = [
    "1 (Not at all)",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7 (Extremely)",
]

# CSAT: satisfacción general, escala 1-5.
CSAT_LABEL = (
    "Overall, how satisfied are you with the assistant?"
)
CSAT_CHOICES = [
    "1 (Very dissatisfied)",
    "2",
    "3 (Neutral)",
    "4",
    "5 (Very satisfied)",
]


# ============================================================
# CAMPOS DE LA ENCUESTA (orden de presentación)
# ============================================================

SURVEY_FIELDS = []

for _item in NASA_TLX_ITEMS:
    SURVEY_FIELDS.append({
        "id": _item["id"],
        "section": "Frustration (NASA-TLX)",
        "label": _item["label"],
        "description": _item["description"],
        "kind": "slider",
        "min": 0,
        "max": 100,
        "step": 1,
    })

for _idx, _item in enumerate(STIAS_ITEMS, start=1):
    SURVEY_FIELDS.append({
        "id": f"stias_{_idx}",
        "section": "Trust in the Assistant (S-TIAS)",
        "label": _item,
        "description": "Rate from 1 (Not at all) to 7 (Extremely).",
        "kind": "radio",
        "choices": STIAS_CHOICES,
    })

SURVEY_FIELDS.append({
    "id": "csat",
    "section": "Satisfaction (CSAT)",
    "label": CSAT_LABEL,
    "kind": "radio",
    "choices": CSAT_CHOICES,
})


# ============================================================
# INTERFAZ GRADIO
# ============================================================

def _make_component(field):
    kind = field["kind"]

    if kind == "slider":
        return gr.Slider(
            minimum=field["min"],
            maximum=field["max"],
            step=field["step"],
            value=field["min"],
            label=field["label"],
            info=field.get("description"),
        )

    if kind == "radio":
        return gr.Radio(
            choices=field["choices"],
            label=field["label"],
            info=field.get("description"),
        )

    raise ValueError(f"Tipo de campo desconocido: {kind}")


def build_survey_block():
    """
    Crea la pantalla de encuesta post-chat.

    Debe llamarse dentro del contexto `with gr.Blocks()`.
    Devuelve un dict con la columna contenedora, los componentes por
    id de campo, el botón de envío y el mensaje de agradecimiento.
    """

    components = {}

    with gr.Column(visible=False) as survey_screen:

        gr.Markdown("# Post-Interaction Survey")
        gr.Markdown(
            "Please answer the following questions about your "
            "interaction with the assistant."
        )

        current_section = None

        for field in SURVEY_FIELDS:
            if field["section"] != current_section:
                current_section = field["section"]
                gr.Markdown(f"## {current_section}")

            components[field["id"]] = _make_component(field)

        submit_button = gr.Button(
            "Submit survey",
            variant="primary",
        )

    with gr.Column(
        visible=False,
        elem_id="thank-you-screen",
    ) as thank_you_screen:

        gr.Markdown(
            "## Thank you for participating, your data was "
            "submitted successfully."
        )

    return {
        "survey_screen": survey_screen,
        "components": components,
        "submit_button": submit_button,
        "thank_you_screen": thank_you_screen,
    }


def submit_survey(session_id, selected_model, *values):
    """
    Registra las respuestas de la encuesta y el fin de sesión.

    `values` se corresponde con los campos de SURVEY_FIELDS en orden.
    Devuelve updates de Gradio: ocultar la encuesta y mostrar la
    pantalla de agradecimiento centrada.
    """

    answers = {}
    for field, value in zip(SURVEY_FIELDS, values):
        answers[field["id"]] = value

    log_event(
        "survey_response",
        session_id=session_id,
        model=selected_model,
        answers=answers,
    )

    log_event(
        "session_end",
        session_id=session_id,
    )

    return (
        gr.update(visible=False),
        gr.update(visible=True),
    )