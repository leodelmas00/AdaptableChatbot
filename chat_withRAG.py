import os
import requests
import gradio as gr

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()


# ============================================================
# GROQ
# ============================================================

def get_groq_models() -> list[str]:
    """
    Llama al endpoint de modelos de Groq y devuelve una lista
    con los IDs de los modelos de solo texto (chat).
    Excluye modelos de audio, TTS y de seguridad.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "La variable de entorno GROQ_API_KEY no está definida"
        )

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

        # Solo modelos texto → texto
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


# Obtener los modelos una sola vez al iniciar
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


# ============================================================
# RAG
# ============================================================

PDF_PATH = "data/SkyRouteTravelAgency.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "skyroute_travel_agency"


def initialize_rag():
    """
    Inicializa el sistema RAG.

    Si la base de Chroma ya existe, se reutiliza.
    Si no existe, se procesa el PDF y se crea.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Si ya existe Chroma, reutilizarla
    if os.path.exists(CHROMA_PATH):

        print("Cargando base de datos Chroma existente...")

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )

    # Primera ejecución
    else:

        print("Creando base de datos Chroma...")

        loader = PyPDFLoader(PDF_PATH)

        documents = loader.load()

        print(f"PDF cargado: {len(documents)} páginas")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documents)

        print(f"Chunks generados: {len(chunks)}")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH
        )

        print("Base de datos Chroma creada.")

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    return retriever


# Inicializar RAG una sola vez al iniciar el programa
retriever = initialize_rag()


# ============================================================
# CHAT
# ============================================================

def chat(message, history, llm, length, formality):
    """
    Procesa el mensaje utilizando:

    - Historial de conversación
    - RAG sobre SkyRouteTravelAgency.pdf
    - Longitud seleccionada
    - Formalidad seleccionada
    """

    # --------------------------------------------------------
    # Recuperar información relevante del PDF
    # --------------------------------------------------------

    documents = retriever.invoke(message)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # --------------------------------------------------------
    # Configuración de la respuesta
    # --------------------------------------------------------

    if length == "Short":
        length_instruction = (
            "Give a short and concise answer. "
            "Avoid unnecessary explanations."
        )
    else:
        length_instruction = (
            "Give a detailed and comprehensive answer. "
            "Provide explanations and relevant context."
        )

    if formality == "Informal":
        formality_instruction = (
            "Use a natural, conversational and informal tone."
        )
    else:
        formality_instruction = (
            "Use a formal, professional and precise tone."
        )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = f"""
You are a helpful assistant.

{length_instruction}

{formality_instruction}

Use the following context from the SkyRoute Travel Agency
document to answer the user's question.

If the answer cannot be found in the provided context,
say that you do not have enough information from the document.

Context:
{context}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # --------------------------------------------------------
    # Historial
    # --------------------------------------------------------

    for item in history:

        messages.append({
            "role": item["role"],
            "content": item["content"]
        })

    # --------------------------------------------------------
    # Mensaje actual
    # --------------------------------------------------------

    messages.append({
        "role": "user",
        "content": message
    })

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    response = llm.invoke(messages)

    return response.content


# ============================================================
# INICIAR CHAT
# ============================================================

def start_chat(selected_model):
    """
    Se ejecuta una sola vez al seleccionar el modelo.

    Crea la instancia de ChatGroq y muestra la interfaz
    principal.
    """

    llm = create_llm(selected_model)

    return (
        gr.update(visible=False),   # Ocultar setup
        gr.update(visible=True),    # Mostrar chat
        selected_model,             # Guardar modelo
        llm,                        # Guardar instancia LLM
        gr.update(value=selected_model)
    )


# ============================================================
# GRADIO
# ============================================================

with gr.Blocks() as demo:

    # --------------------------------------------------------
    # Estado
    # --------------------------------------------------------

    selected_model = gr.State()
    llm = gr.State()

    # --------------------------------------------------------
    # Pantalla inicial
    # --------------------------------------------------------

    with gr.Column(
        visible=True,
        elem_id="setup-screen"
    ) as setup_screen:

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

    # --------------------------------------------------------
    # Pantalla principal
    # --------------------------------------------------------

    with gr.Column(
        visible=False
    ) as main_screen:

        with gr.Row():

            # ------------------------------------------------
            # Configuración
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Chat
            # ------------------------------------------------

            with gr.Column(scale=3):

                chatbot = gr.ChatInterface(
                    fn=chat,
                    additional_inputs=[
                        llm,
                        length,
                        formality
                    ],
                    description="Chatbot con RAG y GRADIO"
                )

    # --------------------------------------------------------
    # Selección del modelo
    # --------------------------------------------------------

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


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    css="""
    #setup-screen {
        max-width: 600px;
        margin: 0 auto;
    }
    """
)