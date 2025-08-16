import os

FILE_NAME = "luatbvdl.txt"
DATA_DIR = "data"
FULL_FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
DB_DIR = "vector_db"
DATABASE_PATH = os.path.join(DB_DIR, "my_knowledge_db")
COLLECTION_NAME = "luat_bao_ve_du_lieu"

EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
CLEAR_EXISTING_DB = False

import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "llama3-8b-8192" 
