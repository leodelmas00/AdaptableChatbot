"""
Sistema RAG basado en ChromaDB para el chatbot experimental.

El PDF de la agencia de viajes se indexa en ChromaDB la primera vez.
Si el índice ya existe, se carga directamente sin reprocesar el PDF.

Para almacenamiento persistente en Hugging Face Spaces, apuntar
CHROMA_PATH a un directorio persistente, por ejemplo /data/chroma_db.
"""

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURACIÓN (configurable por variables de entorno)
# ============================================================

PDF_PATH = os.getenv("PDF_PATH", "data/SkyRouteTravelAgency.pdf")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "skyroute_travel_agency")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

RETRIEVAL_K = int(os.getenv("RAG_K", "4"))

_retriever = None


# ============================================================
# EMBEDDINGS
# ============================================================

def _get_embeddings():
    """
    Instancia los embeddings por modelo. Debe coincidir con el
    modelo usado al crear el índice, o la carga fallará.
    """

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ============================================================
# ÍNDICE
# ============================================================

def _index_exists():
    """
    True si ya existe un índice persistido de ChromaDB.
    """

    return os.path.isfile(os.path.join(CHROMA_PATH, "chroma.sqlite3"))


def _build_index():
    """
    Procesa el PDF, genera los embeddings y crea el índice
    en CHROMA_PATH.
    """

    print("RAG: creando índice ChromaDB desde el PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"RAG: PDF cargado ({len(documents)} páginas)")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"RAG: {len(chunks)} chunks generados")

    embeddings = _get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )

    print(f"RAG: índice creado en {CHROMA_PATH}")

    return vectorstore


def _load_index():
    """
    Carga el índice existente sin reprocesar el PDF.
    """

    print(f"RAG: cargando índice existente desde {CHROMA_PATH}")

    embeddings = _get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )


# ============================================================
# RETRIEVER
# ============================================================

def get_retriever():
    """
    Devuelve el retriever (se carga/crea una sola vez).

    Si el índice existe se carga; si la carga falla (embeddings o
    índice incompatibles) se regenera desde el PDF.
    """

    global _retriever

    if _retriever is not None:
        return _retriever

    vectorstore = None

    if _index_exists():
        try:
            vectorstore = _load_index()
        except Exception as exc:
            print(f"RAG: error al cargar el índice ({exc!r}), regenerando...")
            vectorstore = None

    if vectorstore is None:
        vectorstore = _build_index()

    _retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K},
    )

    return _retriever


def retrieve_context(query):
    """
    Recupera los fragmentos relevantes del PDF para el mensaje dado
    y los devuelve como texto plano (separados por líneas en blanco).
    """

    documents = get_retriever().invoke(query)

    return "\n\n".join(
        document.page_content
        for document in documents
    )