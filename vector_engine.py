import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, DB_PATH, DATA_DIR

CHUNK_CONFIG = {
    # 教授履歷與表格：維持中大容量，但大幅拉高 overlap 防範 Markdown 表格斷頭與標頭遺失
    "professors":  {"chunk_size": 1200, "chunk_overlap": 350},  
    
    # 畢業規定：縮小 size 以維持「單一學年/學期」的資訊純淨度，避免跨年級雜訊干擾
    "111":         {"chunk_size": 600,  "chunk_overlap": 150},  
    "112":         {"chunk_size": 600,  "chunk_overlap": 150},
    "113":         {"chunk_size": 600,  "chunk_overlap": 150},
    "114":         {"chunk_size": 600,  "chunk_overlap": 150},
    
    # 校曆日程：極小切片策略！一行一事件的短句最怕打包，切小才能精準命中月份與特定事件
    "calendar":    {"chunk_size": 350,  "chunk_overlap": 100},  
    
    # 預備防呆
    "default":     {"chunk_size": 600,  "chunk_overlap": 150},
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
            
            cfg = CHUNK_CONFIG.get(sub_dir, CHUNK_CONFIG["default"])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"]
            )
            split = splitter.split_documents(docs)
            all_documents.extend(split)
            print(f"已讀取 [{sub_dir}]，標籤: {tag}，chunk數: {len(split)}")
        except Exception as e:
            print(f"讀取 {sub_dir} 失敗: {e}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")
    
    try:
        vector_db = FAISS.from_documents(all_documents, embeddings)
        vector_db.save_local(DB_PATH)
        print("向量庫建立完成！")
    except Exception as e:
        print(f"建立失敗: {e}")

def initialize_vector_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434")
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

if __name__ == "__main__":
    create_vector_db()