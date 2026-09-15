# AdaptableChatbot

Este repositorio fue creado para la creación y experimentación de un chatbot conversacional adaptable al usuario.

El chatbot utiliza modelos de lenguaje de Groq mediante LangChain y permite seleccionar el modelo a utilizar y configurar características de la respuesta, como su longitud y nivel de formalidad.

## Instalación

Se requiere Python 3.10 o superior.

### 1. Clonar el repositorio

```bash
git clone https://github.com/leodelmas00/AdaptableChatbot.git
cd AdaptableChatbot
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activarlo en Linux/macOS:

```bash
source .venv/bin/activate
```

En Windows:

```bash
.venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

## ⚠️ Consumo de recursos

Durante la instalación de las dependencias o la primera ejecución puede observarse un consumo elevado de CPU. Esto se debe principalmente a las dependencias utilizadas para generar embeddings, como `sentence-transformers`, y a la descarga y carga inicial del modelo de embeddings.

Este comportamiento es normal y debería disminuir una vez completada la instalación y la inicialización.


### 4. Configurar la API de Groq

Crear un archivo `.env` a partir de `.env.example`:

```bash
cp .env.example .env
```

Luego agregar la API Key de Groq:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Ejecutar

Ejecuta cualquiera de los .py presentes, por ejemplo: `chat_noRAG.py`:

```bash
python chat_noRAG.py
```

La aplicación iniciará una interfaz web mediante Gradio.

## Tecnologías

* Python
* Gradio
* LangChain
* Groq
* Chroma
* Hugging Face
* RAG
* PyPDF