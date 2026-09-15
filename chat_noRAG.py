import os
import requests
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_groq_models() -> list[str]:
    """
    Llama al endpoint de modelos de Groq y devuelve una lista
    con los IDs de los modelos de solo texto (chat).
    Excluye modelos de audio, TTS y de seguridad.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("La variable de entorno GROQ_API_KEY no está definida")

    url = "https://api.groq.com/openai/v1/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()

    model_names = []

    for model in data.get("data", []):
        input_mods = model.get("input_modalities", [])
        output_mods = model.get("output_modalities", [])
        model_id = model.get("id", "").lower()
        name = model.get("name", "").lower()

        # Solo modelos de texto → texto
        is_text = (
            "text" in input_mods and
            "text" in output_mods and
            "speech" not in output_mods and
            "transcription" not in output_mods
        )

        # Excluir modelos de seguridad
        is_security = (
            "guard" in model_id or
            "safeguard" in model_id or
            "guard" in name
        )

        if is_text and not is_security:
            model_names.append(model["id"])

    return model_names


# Obtener los modelos una sola vez al iniciar el programa
models = get_groq_models()

default_model = "openai/gpt-oss-20b"

if default_model not in models and models:
    default_model = models[0]


def create_llm(selected_model):
    """
    Crea el ChatGroq una sola vez cuando el usuario
    confirma el modelo.
    """
    return ChatGroq(
        model=selected_model,
        temperature=0,
    )


def chat(message, history, llm, length, formality):
    messages = []

    # Prompt de configuración
    system_prompt = f"""
        You are a helpful assistant.

        Response length: {length}
        Formality: {formality}

        Adapt your response according to these settings.
        """

    messages.append({
        "role": "system",
        "content": system_prompt
    })

    # Historial
    for item in history:
        messages.append({
            "role": item["role"],
            "content": item["content"]
        })

    # Mensaje actual
    messages.append({
        "role": "user",
        "content": message
    })

    response = llm.invoke(messages)

    return response.content


def start_chat(selected_model):
    """
    Se ejecuta una sola vez al seleccionar el modelo.
    Crea el LLM y muestra la interfaz principal.
    """
    llm = create_llm(selected_model)

    return (
        gr.update(visible=False),  # Ocultar pantalla inicial
        gr.update(visible=True),   # Mostrar pantalla principal
        selected_model,            # Guardar modelo seleccionado
        llm,                       # Guardar instancia de ChatGroq
        gr.update(value=selected_model)
    )


with gr.Blocks() as demo:

    # Estado de la aplicación
    selected_model = gr.State()
    llm = gr.State()

    # ─────────────────────────────────────
    # Pantalla inicial
    # ─────────────────────────────────────

    with gr.Column(visible=True, elem_id="setup-screen") as setup_screen:

        gr.Markdown("# Chatbot")

        gr.Markdown("## Select the model")

        model_selector = gr.Dropdown(
            choices=models,
            value=default_model,
            label="Model",
            interactive=True
        )

        continue_button = gr.Button(
            "Continue",
            variant="primary"
        )

    # ─────────────────────────────────────
    # Pantalla principal
    # ─────────────────────────────────────

    with gr.Column(visible=False) as main_screen:

        with gr.Row():

            # Configuración
            with gr.Column(scale=1):

                gr.Markdown("### Configuration")

                model = gr.Dropdown(
                    choices=models,
                    value=default_model,
                    label="Model",
                    interactive=False
                )

                length = gr.Radio(
                    choices=["Short", "Long"],
                    value="Short",
                    label="Length"
                )

                formality = gr.Radio(
                    choices=["Informal", "Formal"],
                    value="Informal",
                    label="Formality"
                )

            # Chat
            with gr.Column(scale=3):

                chatbot = gr.ChatInterface(
                    fn=chat,
                    additional_inputs=[
                        llm,
                        length,
                        formality
                    ],
                    description="Chatbot simple con Groq + LangChain."
                )

    # ─────────────────────────────────────
    # Selección del modelo
    # ─────────────────────────────────────

    continue_button.click(
        fn=start_chat,
        inputs=model_selector,
        outputs=[
            setup_screen,
            main_screen,
            selected_model,
            llm,
            model
        ]
    )


demo.launch(css="""
    #setup-screen {
        max-width: 600px;
        margin: 0 auto;
    }
    """
)