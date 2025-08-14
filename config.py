# config.py
import os

# --- Cấu hình chung ---
# Tên file tài liệu
FILE_NAME = "luatbvdl.txt"
# Đường dẫn tới thư mục chứa các file dữ liệu (data)
DATA_DIR = "data"
# Đường dẫn đầy đủ tới file tài liệu
FULL_FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# --- Cấu hình Vector Database (ChromaDB) ---
# Tên thư mục chứa cơ sở dữ liệu
DB_DIR = "vector_db"
# Tên thư mục cụ thể cho database của bạn
DATABASE_PATH = os.path.join(DB_DIR, "my_knowledge_db")
# Tên của collection trong ChromaDB
COLLECTION_NAME = "luat_bao_ve_du_lieu"

# --- Cấu hình Embedding Model ---
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"

# --- Cấu hình Chunking ---
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# --- Cấu hình khác ---
# Cài đặt để xóa dữ liệu cũ và tạo lại collection mỗi khi build
CLEAR_EXISTING_DB = False

# === Cấu hình Groq API ===
# Thay thế chuỗi này bằng API Key thực tế của bạn từ Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL_NAME = "llama3-8b-8192" # Tên mô hình của Groq.