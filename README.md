# AdaptableChatbot

Este repositorio fue creado para la experimentación con un chatbot conversacional de reservas de viajes para una investigación sobre la forma de las respuestas de chatbots basados en LLM.

El chatbot utiliza modelos de lenguaje de DeepInfra mediante LangChain y permite seleccionar el modelo al inicio de la sesión. Incluye tres condiciones experimentales controladas por configuración.

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

### 4. Configurar la API de DeepInfra

Crear un archivo `.env` a partir de `.env.example`:

```bash
cp .env.example .env
```

Luego agregar la API Key de DeepInfra:

```env
DEEPINFRA_API_KEY=your_deepinfra_api_key_here
```

### 5. Configurar la condición experimental

En `chat_borrador.py`, la variable `CHATBOT_TYPE` define la condición:

* `1` = Chatbot de control (sin módulos de longitud ni formalidad)
* `2` = Chatbot con controles de longitud y formalidad
* `3` = Chatbot con forma predefinida (Long + Formal)

### 6. Ejecutar

```bash
python chat_borrador.py
```

La aplicación iniciará una interfaz web mediante Gradio.

## Tecnologías

* Python
* Gradio
* LangChain
* DeepInfra (API compatible con OpenAI)