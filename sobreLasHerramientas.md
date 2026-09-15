# Herramientas utilizadas

## Gradio

Gradio es una biblioteca de Python que permite crear interfaces web para aplicaciones de inteligencia artificial.

En este proyecto se utiliza como interfaz del chatbot, permitiendo seleccionar el modelo, configurar parámetros de respuesta y mantener la conversación.

Entre los componentes utilizados se encuentran `Blocks`, `Dropdown`, `Radio`, `Button`, `ChatInterface` y `State`. Los eventos permiten conectar las acciones del usuario con funciones de Python.

## LangChain

LangChain es un framework para desarrollar aplicaciones basadas en modelos de lenguaje.

Se utiliza para integrar el modelo de lenguaje con otros componentes de la aplicación, como sistemas RAG, embeddings y bases de datos vectoriales.

También proporciona integraciones con diferentes proveedores de modelos y herramientas relacionadas con LLM.

## Groq

Groq proporciona una API para acceder a modelos de lenguaje.

En este proyecto se utiliza como proveedor del LLM. La aplicación consulta los modelos disponibles y permite seleccionar cuál utilizar durante la conversación.

El acceso se realiza mediante una API Key almacenada como variable de entorno.

## RAG

RAG (Retrieval-Augmented Generation) es una técnica que combina la recuperación de información con la generación de texto mediante un modelo de lenguaje.

De forma general, los documentos se dividen en fragmentos, se generan embeddings y estos se almacenan en una base de datos vectorial. Ante una consulta, se recuperan los fragmentos más relevantes y se incorporan al contexto enviado al modelo.

Esto permite que el modelo utilice información externa y específica de la aplicación al generar sus respuestas.

## Embeddings

Los embeddings son representaciones numéricas del contenido de un texto.

Permiten comparar semánticamente documentos y consultas mediante vectores. En este proyecto se utilizan para representar los fragmentos de documentos que posteriormente son almacenados en una base de datos vectorial.

## Chroma

Chroma es una base de datos vectorial utilizada para almacenar y recuperar embeddings.

En el sistema RAG permite guardar las representaciones de los documentos y recuperar los fragmentos más relevantes ante una consulta.

También permite utilizar almacenamiento persistente, evitando tener que procesar nuevamente los documentos en cada ejecución.

## Hugging Face

Hugging Face proporciona modelos y herramientas para trabajar con inteligencia artificial y procesamiento del lenguaje natural.

En este proyecto se utiliza como fuente de un modelo de embeddings para transformar los textos en representaciones vectoriales.

## PyPDFLoader

`PyPDFLoader` es un componente de LangChain que permite cargar documentos PDF y convertir su contenido en documentos que pueden ser procesados posteriormente.

Puede utilizarse como parte inicial de un flujo RAG para cargar y procesar documentos.

## RecursiveCharacterTextSplitter

`RecursiveCharacterTextSplitter` permite dividir documentos extensos en fragmentos más pequeños.

Esto resulta útil en sistemas RAG, ya que permite recuperar únicamente las partes relevantes de un documento en lugar de utilizarlo completo como contexto.

## Requests

`requests` es una biblioteca de Python para realizar solicitudes HTTP.

En este proyecto se utiliza para comunicarse con APIs externas, como la API del proveedor de modelos.

## python-dotenv

`python-dotenv` permite cargar variables de entorno desde un archivo `.env`.

Se utiliza principalmente para mantener configuraciones sensibles, como las API Keys, fuera del código fuente.

Por ejemplo:

```env
GROQ_API_KEY=your_api_key
```

El archivo `.env` no debe incluirse en el repositorio. En su lugar, puede utilizarse un `.env.example` para indicar las variables necesarias.
