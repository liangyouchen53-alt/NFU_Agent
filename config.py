import os

DATA_DIR = "data/"
DB_PATH = "faiss_index" 
MODEL_NAME = "kenneth85/llama-3-taiwan" 
EMBED_MODEL = "nomic-embed-text" 

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)