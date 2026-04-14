import os

DATA_PATH = "data/"
DB_PATH = "vector_db"
MODEL_NAME = "kenneth85/llama-3-taiwan"

# 確保資料夾存在
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)