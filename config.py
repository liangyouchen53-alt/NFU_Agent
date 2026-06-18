import os

DATA_DIR = "data/"
DB_PATH = "faiss_index" 
#MODEL_NAME = "kenneth85/llama-3-taiwan" 
#MODEL_NAME = "llama3.2:3b"
MODEL_NAME = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text" 

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)