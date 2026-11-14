import streamlit as st
import os
import gdown
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- CONFIGURACIÓN ---
# PEGA AQUÍ EL ID DE TU CARPETA DE DRIVE (ej: 1AbC...xyz)
DRIVE_FOLDER_ID = "TU_ID_DE_CARPETA_AQUI" 
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

st.set_page_config(page_title="Asistente de Documentos", page_icon="📂")

# --- FUNCIONES DE CARGA DE DATOS ---
@st.cache_resource(show_spinner="Cargando datos desde Drive...")
def cargar_datos_drive():
    # Crear carpeta temporal
    if os.path.exists("./data"):
        shutil.rmtree("./data")
    os.makedirs("./data")
    
    # Descargar carpeta de Drive
    url = f"https://drive.google.com/drive/folders/1hrZLnqVqmsUz0n0FwKCrDJ7XQOILrO42"
    try:
        gdown.download_folder(url, output="./data", quiet=False, use_cookies=False)
        return True
    except Exception as e:
        st.error(f"Error al descargar de Drive: {e}")
        return False

@st.cache_resource(show_spinner="Procesando documentos...")
def procesar_documentos():
    loader = DirectoryLoader("./data", glob="**/*.*", use_multithreading=True)
    docs = loader.load()
    
    if not docs:
        return None

    # Dividir texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Crear base de datos vectorial (Memoria)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# --- INTERFAZ ---
st.title("🤖 Asistente Restringido por Documentos")
st.caption("Este bot solo responde basándose en los archivos cargados.")

if not GROQ_API_KEY:
    st.error("⚠️ Falta la API Key de Groq. Configúrala en los Secrets de Streamlit.")
    st.stop()

# Inicializar sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cargar datos
with st.spinner("Inicializando sistema..."):
    if cargar_datos_drive():
        vectorstore = procesar_documentos()
        if not vectorstore:
            st.warning("No se encontraron archivos en la carpeta de Drive.")
            st.stop()
    else:
        st.stop()

# Configurar LLM (Groq es gratuito y rápido)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)

# Crear cadena de conversación con restricción
# El "System Prompt" es clave para que SOLO responda preguntas determinadas
qa_prompt = """Eres un asistente experto en la información contenida en los documentos proporcionados.
REGLA DE ORO: Solo puedes responder preguntas basándote EXCLUSIVAMENTE en el contexto de los documentos.
Si la respuesta no está en el contexto o te preguntan algo fuera de tema (como el clima, opiniones políticas, o chistes), 
debes responder exactamente: "Lo siento, solo puedo responder preguntas relacionadas con los documentos suministrados."
No inventes información.
Contexto: {context}
Pregunta: {question}
Respuesta útil:"""

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template=qa_prompt, input_variables=["context", "question"])

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Busca en los 3 fragmentos más relevantes
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Chat History
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt_user := st.chat_input("¿Qué quieres saber sobre los documentos?"):
    st.session_state.messages.append({"role": "user", "content": prompt_user})
    with st.chat_message("user"):
        st.markdown(prompt_user)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                result = qa_chain.invoke({
                    "question": prompt_user,
                    "chat_history": st.session_state.get("chat_history", [])
                })
                response = result["answer"]
                
                # Actualizar historial para la memoria
                st.session_state.chat_history = result.get("chat_history", [])
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
