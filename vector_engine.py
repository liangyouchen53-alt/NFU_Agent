# vector_engine.py
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from config import DATA_PATH, DB_PATH, MODEL_NAME

def initialize_vector_db():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print("⚠️ 提示：data 資料夾內沒有 PDF。")
        return None

    print("📖 正在優化 PDF 索引...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=MODEL_NAME),
        persist_directory=DB_PATH
    )
    return vector_db