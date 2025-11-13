# indexer.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_together import TogetherEmbeddings

# Carga las variables de entorno (.env)
load_dotenv()

# Rutas y Modelos
DATA_PATH = "docs" 
DB_PATH = "chroma_db"
# Modelo de embeddings de Together.ai (gratuito o eficiente)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" 

def create_vector_db():
    # Verificación inicial de documentos
    if not any(fname.endswith('.pdf') for fname in os.listdir(DATA_PATH)):
        print(f"ERROR: No se encontraron archivos PDF en la carpeta '{DATA_PATH}'.")
        print("Por favor, coloca tus documentos de Lean Management (PDFs) allí y vuelve a intentar.")
        return

    # 1. Cargar documentos
    print("Cargando documentos desde 'docs/'...")
    loader = DirectoryLoader(DATA_PATH, glob='**/*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    # 2. Dividir documentos en trozos (chunks)
    print(f"Documentos cargados: {len(documents)} páginas.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(texts)} trozos para el RAG.")

    # 3. Definir el modelo de embeddings usando Together.ai
    print("Inicializando modelo de Embeddings...")
    embeddings = TogetherEmbeddings(model=EMBEDDING_MODEL, 
                                    together_api_key=os.getenv("TOGETHER_API_KEY"))

    # 4. Crear y persistir la base de datos vectorial (ChromaDB)
    print("Creando Base de Datos Vectorial...")
    # Limpiamos si ya existe para evitar errores
    if os.path.exists(DB_PATH):
         import shutil
         shutil.rmtree(DB_PATH)
         print("Base de datos anterior eliminada.")

    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    db.persist()
    print(f"✅ ¡Base de datos vectorial creada y lista en '{DB_PATH}'!")

if __name__ == "__main__":
    create_vector_db() 
