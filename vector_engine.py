import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, DB_PATH, DATA_DIR

def create_vector_db():
    all_documents = []
    if not os.path.exists(DATA_DIR):
        print(f"❌ 找不到資料夾：{DATA_DIR}")
        return

    sub_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for sub_dir in sub_dirs:
        folder_path = os.path.join(DATA_DIR, sub_dir)
        tag = "roger" if sub_dir == "roger" else ("all" if sub_dir == "calendar" else sub_dir)
        
        try:
            # ✅ 使用 TextLoader 並強制 utf-8 編碼，防止中文讀取錯誤導致數字亂跳
            loader = DirectoryLoader(
                folder_path, 
                glob="**/*.md", 
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["year"] = tag
            all_documents.extend(docs)
            print(f"✅ 已讀取 [{sub_dir}]，標籤: {tag}")
        except Exception as e:
            print(f"❌ 讀取 {sub_dir} 失敗: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_documents)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")
    
    try:
        vector_db = FAISS.from_documents(split_docs, embeddings)
        vector_db.save_local(DB_PATH)
        print("🚀 向量庫建立完成，妥當啦！")
    except Exception as e:
        print(f"❌ 建立失敗: {e}")

def initialize_vector_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

if __name__ == "__main__":
    create_vector_db()