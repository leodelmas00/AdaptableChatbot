import os
import time
import uuid
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import formality as formality_clf
import rag
import surveys
from metrics import (
    check_formality_compliance,
    check_length_compliance,
    hedge_density,
    log_event,
    sentence_count,
    word_count,
)


load_dotenv()


# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================

# 1 = Solo chat
# 2 = Chat + controles de longitud/formalidad
# 3 = Chat modificado: Long + Formal
CHATBOT_TYPE = int(os.getenv("CHATBOT_TYPE", "1"))


# ============================================================
# MODELOS
# ============================================================

MODELS = [
    "deepseek-ai/DeepSeek-V4-Pro",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
]

MODEL_LABELS = {
    "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8":
        "Llama 4 Maverick",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506":
        "Mistral Small 3.2",
}

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"


# ============================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================

if CHATBOT_TYPE not in [1, 2, 3]:
    raise ValueError("CHATBOT_TYPE debe ser 1, 2 o 3")


# ============================================================
# CREACIÓN DEL MODELO
# ============================================================

def create_llm(selected_model):
    """
    Crea una instancia de ChatOpenAI apuntando a DeepInfra.

    DeepInfra utiliza una API compatible con OpenAI.
    """

    api_key = os.getenv("DEEPINFRA_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable de entorno DEEPINFRA_API_KEY "
            "no está definida"
        )

    return ChatOpenAI(
        model=selected_model,
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
        temperature=0,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
    )


# ============================================================
# PROMPTS
# ============================================================

BASE_PROMPT = """[BASE]
You are a conversational assistant answering user queries about
travel planning. Respond in English."""


LENGTH_SHORT_PROMPT = """[LENGTH MODULE - short]
Respond in between 30 and 60 words total; aim for approximately 45
words. Do not add information beyond what is strictly requested, even
if other modules suggest further development."""


LENGTH_LONG_DEFINITION = """[LENGTH MODULE - long]
Respond in between 140 and 200 words total; aim for approximately 170
words."""


FORMALITY_FORMAL_PROMPT = """[FORMALITY MODULE - formal]
Formal register: write out every word in full (use "do not" and "it is"
in their complete forms). Use precise terminology, careful syntax, and
impersonal or third-person constructions throughout. Maintain an
elevated, professional tone in every sentence."""


FORMALITY_INFORMAL_PROMPT = """[FORMALITY MODULE - informal]
Informal register: contractions (’don’t’, ’it’s’), everyday vocabulary,
second-person address (’you’). Avoid unnecessary technical terms when a
common equivalent exists. Allowed: moderate interjections (’sure’,
’look’)."""


ANTI_INTERFERENCE_GUARDRAIL = """[ANTI-INTERFERENCE GUARDRAIL]
The 2 dimensions (length, formality) are INDEPENDENT. Satisfy BOTH
SIMULTANEOUSLY without one relaxing the other. Specifically:
- A short response can be fully informal without exceeding the word
  limit.
- A formal response can be short.
- A long response reaches the SAME word count in either register.
Casual phrasing is not a reason to finish early, and formal phrasing
is not a reason to run past the limit. Hit the stated target either
way.
- Register never justifies going over or under the word count, and the
  word count never justifies dropping the register.
Before responding, verify internally that both conditions are met."""


SCENARIO_PROMPT = """[SCENARIO]
You are a flight-booking assistant. Stay strictly on flights and travel:
greet briefly and help the user plan their trip. If the user brings up
anything unrelated to flights or travel, politely decline and steer the
conversation back to their trip.
However, the flight-booking system is currently unreachable, so you cannot
search for, check, or book any flights. Whenever the user asks to find,
check, or book a flight, clearly say you cannot reach the booking system
right now, and offer the SAME workaround every time: suggest they try again
later or check the airline's website directly. Never invent flights,
prices, schedules, or availability."""


# ============================================================
# REFINEMENTS POR MODELO
# ============================================================

INFORMAL_REFINEMENT_SHARED = """You are texting a friend back. Not advising them, not helping them -
just replying. Speak from your own experience: say what you did or
what you’d do, never what "one can" or "you should" do."""


FORMAL_LONG_REFINEMENT = """A formal answer here is
economical, not padded: make your point in no more than 9 sentences
and stop."""


MODEL_REFINEMENTS = {

    "deepseek-ai/DeepSeek-V4-Pro": {
        "formality": (
            INFORMAL_REFINEMENT_SHARED
            + """
Start at least one sentence with "and", "but", "so", "also",
"honestly", or "yeah", and include at least one sentence fragment."""
        ),
        "length": """Keep the whole answer under 180 words; aim for about 165.
Stop once the question is covered, even if more could be said."""
    },

    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "formality": (
            INFORMAL_REFINEMENT_SHARED
            + """
Include at least one sentence fragment."""
        ),
        "length": """Develop the topic in depth with examples where applicable."""
    },

    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
        "formality": (
            INFORMAL_REFINEMENT_SHARED
            + """
Start at least three sentences with "and", "but", "so",
"also", "honestly", "i mean", or "yeah". Include at least one aside
in the middle of a sentence, and at least one sentence fragment."""
        ),
        "length": """Develop the topic in depth with examples where applicable.""",
        "formal": FORMAL_LONG_REFINEMENT,
    },
}


# ============================================================
# CONSTRUCCIÓN DEL SYSTEM PROMPT
# ============================================================

def build_system_prompt(
    selected_model,
    chatbot_type,
    length,
    formality,
    query_text
):
    """
    Construye el prompt experimental respetando los bloques definidos
    en el artículo.

    chatbot_type:
        1 = BASE + TASK
        2 = BASE + LENGTH + FORMALITY + GUARDRAIL + TASK
        3 = BASE + LONG + FORMAL + GUARDRAIL + TASK
    """

    # --------------------------------------------------------
    # CHATBOT 1
    # --------------------------------------------------------

    if chatbot_type == 1:

        return f"""{BASE_PROMPT}

[TASK]
{query_text}"""


    # --------------------------------------------------------
    # CHATBOT 2
    # --------------------------------------------------------

    if chatbot_type == 2:

        # LENGTH
        if length == "Short":
            length_block = LENGTH_SHORT_PROMPT

        else:
            length_refinement = MODEL_REFINEMENTS[selected_model]["length"]

            length_block = f"""{LENGTH_LONG_DEFINITION} {length_refinement}"""


        # FORMALITY
        if formality == "Formal":

            formality_block = f"""{FORMALITY_FORMAL_PROMPT}"""

            # Según blocks.yaml, el refinement formal se aplica
            # únicamente al nivel Long y solo a Mistral.
            if length == "Long":
                formal_refinement = MODEL_REFINEMENTS[
                    selected_model
                ].get("formal", "")

                if formal_refinement:
                    formality_block += f" {formal_refinement}"

        else:

            formality_refinement = MODEL_REFINEMENTS[
                selected_model
            ]["formality"]

            formality_block = f"""{FORMALITY_INFORMAL_PROMPT}
{formality_refinement}"""


        return f"""{BASE_PROMPT}

{length_block}

{formality_block}

{ANTI_INTERFERENCE_GUARDRAIL}

[TASK]
{query_text}"""


    # --------------------------------------------------------
    # CHATBOT 3
    # --------------------------------------------------------

    if chatbot_type == 3:

        length_refinement = MODEL_REFINEMENTS[selected_model]["length"]

        length_block = f"""{LENGTH_LONG_DEFINITION} {length_refinement}"""

        formality_block = f"""{FORMALITY_FORMAL_PROMPT}"""

        # Según blocks.yaml, el refinement formal se aplica solo a Mistral.
        formal_refinement = MODEL_REFINEMENTS[selected_model].get("formal", "")

        if formal_refinement:
            formality_block += f"\n{formal_refinement}"


        return f"""{BASE_PROMPT}

{length_block}

{formality_block}

{ANTI_INTERFERENCE_GUARDRAIL}

[TASK]
{query_text}"""


# ============================================================
# CHAT
# ============================================================

def chat(
    message,
    history,
    llm,
    selected_model,
    length,
    formality,
    session_id
):
    """
    Procesa el mensaje del usuario combinando cuatro fuentes:

    - Prompt experimental (sistema), según CHATBOT_TYPE/longitud/formalidad.
    - Escenario de sistema de reservas inalcanzable (copiado de blocks.yaml).
    - Contexto RAG recuperado de ChromaDB desde el PDF de la agencia.
    - Historial conversacional completo de la sesión actual (memoria).

    Además de devolver la respuesta, registra las métricas del turno
    (timestamp, latencia, errores, etc.) de forma desacoplada.
    """

    system_prompt = build_system_prompt(
        selected_model=selected_model,
        chatbot_type=CHATBOT_TYPE,
        length=length,
        formality=formality,
        query_text=message,
    )

    # Contexto RAG: conocimiento externo recuperado del PDF.
    # Si la recuperación falla, se continúa sin contexto.
    try:
        rag_context = rag.retrieve_context(message)
    except Exception:
        rag_context = ""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    # Escenario: sistema de reservas inalcanzable (idéntico en las 3
    # condiciones, copiado de blocks.yaml).
    messages.append({
        "role": "system",
        "content": SCENARIO_PROMPT,
    })

    if rag_context:
        messages.append({
            "role": "system",
            "content": (
                "Relevant information from the travel agency "
                "documentation:\n\n"
                + rag_context
            ),
        })

    # Memoria conversacional: historial completo de la sesión actual.
    # El historial visual de Gradio ya incluye los turnos previos;
    # aquí sí se reenvían al LLM para dar contexto conversacional.
    for item in history:
        if (
            isinstance(item, dict)
            and item.get("role")
            and item.get("content") is not None
        ):
            messages.append({
                "role": item["role"],
                "content": item["content"],
            })

    messages.append({
        "role": "user",
        "content": message,
    })

    # Valores efectivos según la condición experimental
    if CHATBOT_TYPE == 1:
        effective_length = None
        effective_formality = None
    elif CHATBOT_TYPE == 3:
        effective_length = "Long"
        effective_formality = "Formal"
    else:
        effective_length = length
        effective_formality = formality

    # Un turno = un mensaje del usuario + la respuesta del asistente.
    # En Gradio 6 el historial es una lista de mensajes estilo OpenAI
    # (rol usuario/asistente), por lo que solo cuentan los "user".
    turn = 1 + sum(
        1 for m in history
        if isinstance(m, dict) and m.get("role") == "user"
    )

    start = time.perf_counter()
    error = None
    error_type = None
    response = None

    try:
        result = llm.invoke(messages)
        response = result.content
    except Exception as exc:
        error = repr(exc)
        error_type = type(exc).__name__

    latency_ms = (time.perf_counter() - start) * 1000

    # Métricas lingüísticas sobre la respuesta generada
    if response is not None:
        resp_words = word_count(response)
        resp_sentences = sentence_count(response)
        hedge_den = hedge_density(response)

        try:
            formality_label, formality_score = (
                formality_clf.classify_formality(response)
            )
        except Exception:
            formality_label = None
            formality_score = None
    else:
        resp_words = None
        resp_sentences = None
        hedge_den = None
        formality_label = None
        formality_score = None

    log_event(
        "turn",
        session_id=session_id,
        chatbot_type=CHATBOT_TYPE,
        model=selected_model,
        turn=turn,
        length=effective_length,
        formality=effective_formality,
        user_message=message,
        response=response,
        latency_ms=latency_ms,
        error=error,
        error_type=error_type,
        word_count=resp_words,
        sentence_count=resp_sentences,
        hedge_density=hedge_den,
        formality_label=formality_label,
        formality_score=formality_score,
        length_compliant=check_length_compliance(
            resp_words, effective_length
        ),
        formality_compliant=check_formality_compliance(
            formality_label, effective_formality
        ),
    )

    if error is not None:
        return "I'm sorry, something went wrong. Please try again."

    return response


# ============================================================
# INICIO DE SESIÓN
# ============================================================

def start_chat(selected_model):
    """
    Se ejecuta una sola vez al seleccionar el modelo.

    El modelo queda bloqueado durante toda la sesión.
    Se genera un session_id único para agrupar los turnos.
    """

    llm = create_llm(selected_model)
    session_id = uuid.uuid4().hex

    log_event(
        "session_start",
        session_id=session_id,
        model=selected_model,
        chatbot_type=CHATBOT_TYPE,
    )

    return (
        gr.update(visible=False),       # Ocultar setup
        gr.update(visible=True),        # Mostrar chat
        selected_model,                 # Guardar modelo
        llm,                            # Guardar instancia
        gr.update(
            value=selected_model
        ),
        session_id,                     # Guardar id de sesión
    )


# ============================================================
# CAMBIOS DE CONFIGURACIÓN
# ============================================================

def on_length_change(length, session_id):
    """
    Registra un cambio del control de longitud (Chatbot 2).
    """

    log_event(
        "config_change",
        session_id=session_id,
        field="length",
        new_value=length,
    )


def on_formality_change(formality, session_id):
    """
    Registra un cambio del control de formalidad (Chatbot 2).
    """

    log_event(
        "config_change",
        session_id=session_id,
        field="formality",
        new_value=formality,
    )


def finish_chat():
    """
    Oculta el chat y muestra la encuesta post-interacción.
    """

    return (
        gr.update(visible=False),
        gr.update(visible=True),
    )


# ============================================================
# INTERFAZ
# ============================================================

with gr.Blocks() as demo:

    # --------------------------------------------------------
    # ESTADO DE LA APLICACIÓN
    # --------------------------------------------------------

    selected_model = gr.State()
    llm = gr.State()
    session_id = gr.State()


    # --------------------------------------------------------
    # PANTALLA INICIAL
    # --------------------------------------------------------

    with gr.Column(
        visible=True,
        elem_id="setup-screen"
    ) as setup_screen:

        gr.Markdown("# Chatbot")

        gr.Markdown("## Select the model")

        model_selector = gr.Dropdown(
            choices=[
                (MODEL_LABELS[model], model)
                for model in MODELS
            ],
            value=DEFAULT_MODEL,
            label="Model",
            interactive=True,
        )

        continue_button = gr.Button(
            "Continue",
            variant="primary",
        )


    # --------------------------------------------------------
    # PANTALLA PRINCIPAL
    # --------------------------------------------------------

    with gr.Column(
        visible=False
    ) as main_screen:

        with gr.Row(elem_id="main-row"):

            # ------------------------------------------------
            # CONFIGURACIÓN
            # ------------------------------------------------

            with gr.Column(
                scale=1,
                elem_id="config-column",
            ):

                model = gr.Dropdown(
                    choices=[
                        (MODEL_LABELS[m], m)
                        for m in MODELS
                    ],
                    value=DEFAULT_MODEL,
                    label="Model",
                    interactive=False,
                )

                if CHATBOT_TYPE == 2:

                    length = gr.Radio(
                        choices=["Short", "Long"],
                        value="Short",
                        label="Length",
                    )

                    formality = gr.Radio(
                        choices=["Informal", "Formal"],
                        value="Informal",
                        label="Formality",
                    )

                    length.change(
                        fn=on_length_change,
                        inputs=[length, session_id],
                        outputs=[],
                    )

                    formality.change(
                        fn=on_formality_change,
                        inputs=[formality, session_id],
                        outputs=[],
                    )

                elif CHATBOT_TYPE == 1:
                    length = gr.State(value="Short")
                    formality = gr.State(value="Informal")

                else:
                    length = gr.State(value="Long")
                    formality = gr.State(value="Formal")

                gr.HTML("<div style='flex-grow: 1;'></div>")

                finish_button = gr.Button(
                    "Finish",
                    variant="secondary",
                )


            # ------------------------------------------------
            # CHAT
            # ------------------------------------------------

            with gr.Column(scale=3):

                gr.ChatInterface(
                    fn=chat,
                    additional_inputs=[
                        llm,
                        selected_model,
                        length,
                        formality,
                        session_id,
                    ],
                    
                )


    # --------------------------------------------------------
    # ENCUESTA POST-INTERACCIÓN
    # --------------------------------------------------------

    survey_ui = surveys.build_survey_block()


    # --------------------------------------------------------
    # SELECCIÓN DEL MODELO
    # --------------------------------------------------------

    continue_button.click(
        fn=start_chat,
        inputs=model_selector,
        outputs=[
            setup_screen,
            main_screen,
            selected_model,
            llm,
            model,
            session_id,
        ],
    )


    # --------------------------------------------------------
    # FINALIZAR CHAT Y ENCUESTA
    # --------------------------------------------------------

    finish_button.click(
        fn=finish_chat,
        inputs=[],
        outputs=[
            main_screen,
            survey_ui["survey_screen"],
        ],
    )

    survey_ui["submit_button"].click(
        fn=surveys.submit_survey,
        inputs=[
            session_id,
            selected_model,
        ] + [
            survey_ui["components"][field["id"]]
            for field in surveys.SURVEY_FIELDS
        ],
        outputs=[
            survey_ui["survey_screen"],
            survey_ui["thank_you_screen"],
        ],
    )


# ============================================================
# EJECUTAR
# ============================================================

demo.launch(
    css="""
        #setup-screen {
            max-width: 600px;
            margin: 0 auto;
        }
        #main-row {
            align-items: stretch;
        }
        #config-column {
            display: flex;
            flex-direction: column;
        }
        #thank-you-screen {
            min-height: 80vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
    """
)