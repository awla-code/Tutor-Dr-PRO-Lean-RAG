 
# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_together import TogetherEmbeddings
from langchain_together import Together
from langchain.chains import ConversationalRetrievalChain

# Carga las variables de entorno
load_dotenv()

# --- Configuración de la IA ---
DB_PATH = "chroma_db"
EMBED_MODEL = "BAAI/bge-large-en-v1.5" 
# El modelo Llama 3 70B gratuito
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" 

#limpiar Caracteres No-ASCII de la Pregunta
import unicodedata

def clean_text_for_api(text):
    """Remueve caracteres problematicos como el signo de interrogación de apertura."""
    # Se asegura de que la codificación sea UTF-8 y luego la decodifica
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    # Remueve el signo de interrogación de apertura y otros caracteres no-ASCII si persisten
    cleaned_text = text.replace('¿', '').replace('¡', '')
    return cleaned_text

# Inicializar los componentes de LangChain con caché
@st.cache_resource
def setup_rag_components():
    """Inicializa el LLM, embeddings y la cadena RAG."""
    if not os.path.exists(DB_PATH):
        st.error("ERROR: No se encontró la base de datos 'chroma_db'. ¿Ejecutaste indexer.py primero?")
        st.stop()

    # 1. Cargar el LLM (Llama 3 70B)
    llm = Together(
        model=LLM_MODEL,
        temperature=0.7,
        together_api_key=os.getenv("TOGETHER_API_KEY")
    )

    # 2. Cargar Embeddings
    embeddings = TogetherEmbeddings(model=EMBED_MODEL, 
                                    together_api_key=os.getenv("TOGETHER_API_KEY"))

    # 3. Cargar Base de Datos Vectorial
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 4. Crear la cadena RAG conversacional (mantiene el historial)
    # system_prompt ayuda a definir la personalidad del tutor
    system_prompt = (
        "Eres Dr. PRO, un tutor IA experto en Lean Management. "
        "Tu objetivo es responder preguntas de manera clara, concisa y usando los documentos de "
        "Lean Management que te proporcionaron. Manten un tono profesional y motivador."
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False 
    )

    # Inyectamos el system prompt para guiar el LLM (esto mejora el comportamiento)
    def custom_chain(data):
        # APLICAMOS LA LIMPIEZA DEL PROMPT ANTES DE USARLO
        prompt_limpio = clean_text_for_api(data["question"]) 
        
        # El formato LLama 3 se aplica internamente en langchain_together/llms
        # Aquí sólo pasamos el prompt limpio y el historial
        
        # Usamos el prompt limpio para la recuperación (embedding) y para la llamada final
        return chain.invoke({"question": prompt_limpio, "chat_history": data["chat_history"]})

    return custom_chain # <--- Esto es lo que retorna setup_rag_components

# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Dr. PRO: Tutor IA Lean Management", layout="wide")
st.title("Dr. PRO: Tu Tutor IA de Lean Management")
st.subheader("Impulsa tu productividad con IA gratuita y open-source.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hola, soy **Dr. PRO**, tu tutor IA especializado. Pregúntame sobre 5S, Kanban, Kaizen o cualquier concepto que necesites aprender para mejorar tu productividad."}
    ]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Inicializar la cadena RAG
try:
    qa_chain = setup_rag_components()
except Exception as e:
    st.error(f"Error al configurar la IA: {e}")
    st.stop()


# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar y procesar la entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta sobre Lean Management..."):
    # Agregar mensaje de usuario al historial de la UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Llamar a la cadena RAG
    with st.chat_message("assistant"):
        with st.spinner("Dr. PRO está pensando..."):
            # Adaptamos la llamada a la función personalizada (custom_chain)
            result = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
            response = result["answer"]
            st.markdown(response)

    # Actualizar historial de chat y de la UI
    st.session_state.chat_history.append((prompt, response))
    st.session_state.messages.append({"role": "assistant", "content": response})