import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, DB_PATH, DATA_DIR

# 🔧 修改一：針對不同資料夾設定不同的 chunk_size
# professors.md 是表格型資料，切太小會破壞一筆教授資料的完整性
CHUNK_CONFIG = {
    "professors":  {"chunk_size": 2000, "chunk_overlap": 100},  # 教授表格，保持完整一筆
    "111":         {"chunk_size": 1000,  "chunk_overlap": 150},  # 畢業規定，數字不能切斷
    "112":         {"chunk_size": 1000,  "chunk_overlap": 150},
    "113":         {"chunk_size": 1000,  "chunk_overlap": 150},
    "114":         {"chunk_size": 1000,  "chunk_overlap": 150},
    "default":     {"chunk_size": 1000,  "chunk_overlap": 150},
}

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
            loader = DirectoryLoader(
                folder_path, 
                glob="**/*.md", 
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["year"] = tag
            
            # 🔧 修改一：依資料夾套用不同切割設定
            cfg = CHUNK_CONFIG.get(sub_dir, CHUNK_CONFIG["default"])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"]
            )
            split = splitter.split_documents(docs)
            all_documents.extend(split)
            print(f"✅ 已讀取 [{sub_dir}]，標籤: {tag}，chunk數: {len(split)}")
        except Exception as e:
            print(f"❌ 讀取 {sub_dir} 失敗: {e}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")
    
    try:
        vector_db = FAISS.from_documents(all_documents, embeddings)
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