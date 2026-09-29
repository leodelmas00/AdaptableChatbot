import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================

# 1 = Solo chat
# 2 = Chat + controles de longitud/formalidad
# 3 = Chat modificado: Long + Formal
CHATBOT_TYPE = 1


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


# ============================================================
# REFINEMENTS POR MODELO
# ============================================================

INFORMAL_REFINEMENT_SHARED = """You are texting a friend back. Not advising them, not helping them -
just replying. Speak from your own experience: say what you did or
what you’d do, never what "one can" or "you should" do."""


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
        "length": """Develop the topic in depth with examples where applicable."""
    },
}


FORMAL_LONG_REFINEMENT = """A formal answer here is
economical, not padded: make your point in no more than 9 sentences
and stop."""


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

            # Según el artículo, el refinement formal se aplica
            # únicamente al nivel Long.
            if length == "Long":
                formality_block += f" {FORMAL_LONG_REFINEMENT}"

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

        formality_block = f"""{FORMALITY_FORMAL_PROMPT}
{FORMAL_LONG_REFINEMENT}"""


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
    formality
):
    """
    Envía solamente el mensaje actual al modelo.

    El historial visual de Gradio NO se envía al LLM.
    Por lo tanto, el chatbot no tiene memoria conversacional.
    """

    system_prompt = build_system_prompt(
        selected_model=selected_model,
        chatbot_type=CHATBOT_TYPE,
        length=length,
        formality=formality,
        query_text=message,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    response = llm.invoke(messages)

    return response.content


# ============================================================
# INICIO DE SESIÓN
# ============================================================

def start_chat(selected_model):
    """
    Se ejecuta una sola vez al seleccionar el modelo.

    El modelo queda bloqueado durante toda la sesión.
    """

    llm = create_llm(selected_model)

    return (
        gr.update(visible=False),       # Ocultar setup
        gr.update(visible=True),        # Mostrar chat
        selected_model,                 # Guardar modelo
        llm,                            # Guardar instancia
        gr.update(
            value=MODEL_LABELS[selected_model]
        ),
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

        with gr.Row():

            # ------------------------------------------------
            # CONFIGURACIÓN
            # ------------------------------------------------

            with gr.Column(scale=1):

                gr.Markdown("### Configuration")

                model = gr.Dropdown(
                    choices=[
                        (MODEL_LABELS[m], m)
                        for m in MODELS
                    ],
                    value=DEFAULT_MODEL,
                    label="Model",
                    interactive=False,
                )


                # ============================================
                # CHATBOT 1
                # ============================================

                if CHATBOT_TYPE == 1:

                    length = gr.Radio(
                        choices=["Short", "Long"],
                        value="Short",
                        label="Length",
                        visible=False,
                    )

                    formality = gr.Radio(
                        choices=["Informal", "Formal"],
                        value="Informal",
                        label="Formality",
                        visible=False,
                    )

                    gr.Markdown(
                        "No response preferences are configured."
                    )


                # ============================================
                # CHATBOT 2
                # ============================================

                elif CHATBOT_TYPE == 2:

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


                # ============================================
                # CHATBOT 3
                # ============================================

                elif CHATBOT_TYPE == 3:

                    length = gr.Radio(
                        choices=["Short", "Long"],
                        value="Long",
                        label="Length",
                        visible=False,
                    )

                    formality = gr.Radio(
                        choices=["Informal", "Formal"],
                        value="Formal",
                        label="Formality",
                        visible=False,
                    )

                    gr.Markdown(
                        "Response style: Long + Formal"
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
                    ],
                    description=(
                        "Travel planning assistant. "
                        "The system is currently unavailable."
                    ),
                )


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
    """
)