# build_kb.py
import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import chromadb

# Import các module đã tạo
from config import FULL_FILE_PATH, DATABASE_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, \
    CLEAR_EXISTING_DB
from text_processor import TextProcessor
from embedding_generator import EmbeddingGenerator
from vector_database import VectorDatabase

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """Lớp chính để xây dựng kho tri thức"""

    def __init__(self, db_path: str, collection_name: str, embedding_model: str):
        self.text_processor = TextProcessor()
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_db = VectorDatabase(db_path, collection_name)
        logger.info("🚀 KnowledgeBaseBuilder đã khởi tạo thành công.")

    def build_from_file(self, file_path: str, chunk_size: int, overlap: int, clear_existing: bool):
        logger.info(f"🔄 Bắt đầu xây dựng knowledge base từ file: {file_path}")

        # Xóa dữ liệu cũ nếu được cấu hình
        if clear_existing:
            logger.info(f"🔥 Yêu cầu xóa dữ liệu cũ. Đang tạo lại collection '{self.vector_db.collection_name}'...")
            try:
                self.vector_db.client.delete_collection(name=self.vector_db.collection_name)
                self.vector_db.collection = self.vector_db.client.create_collection(name=self.vector_db.collection_name)
                logger.info(f"✅ Đã tạo lại collection '{self.vector_db.collection_name}' thành công.")
            except Exception as e:
                logger.error(f"❌ Lỗi khi tạo lại collection: {e}. Có thể collection không tồn tại.")
                # Nếu lỗi, thử lại việc get_or_create để đảm bảo collection tồn tại
                self.vector_db.collection = self.vector_db.client.get_or_create_collection(
                    name=self.vector_db.collection_name)

        raw_text = self.text_processor.read_file(file_path)
        if not raw_text: return False

        cleaned_text = self.text_processor.clean_text(raw_text)
        if not cleaned_text: return False

        chunks = self.text_processor.split_into_chunks(cleaned_text, chunk_size, overlap)
        if not chunks: return False

        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_generator.create_embeddings(texts)
        if not embeddings: return False

        success = self.vector_db.add_documents(chunks, embeddings, Path(file_path).name)
        if success:
            logger.info("🎉 Xây dựng knowledge base thành công!")
            self.print_summary()
            return True
        return False

    def print_summary(self):
        db_info = self.vector_db.get_database_info()
        model_info = self.embedding_generator.get_model_info()
        print("\n" + "=" * 60 + "\n📊 TỔNG KẾT KNOWLEDGE BASE\n" + "=" * 60)
        print(f"📍 Đường dẫn DB: {db_info.get('database_path', 'N/A')}")
        print(f"📚 Collection: {db_info.get('collection_name', 'N/A')}")
        print(f"📄 Tổng số documents: {db_info.get('total_documents', 0)}")
        print(f"🤖 Embedding model: {model_info.get('model_name', 'N/A')}")
        print(f"📐 Vector dimension: {model_info.get('vector_dimension', 'N/A')}")
        print("=" * 60 + "\n")


# --- Phần thực thi chính ---
if __name__ == "__main__":
    print("🚀 BẮT ĐẦU XÂY DỰNG KNOWLEDGE BASE")
    print("-" * 50)

    try:
        builder = KnowledgeBaseBuilder(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL_NAME
        )
        success = builder.build_from_file(
            file_path=FULL_FILE_PATH,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            clear_existing=CLEAR_EXISTING_DB
        )

        if success:
            print("Xây dựng knowledge base thành công!")
        else:
            print("Có lỗi xảy ra trong quá trình xây dựng")

    except Exception as e:
        logger.error(f"FATAL ERROR trong quá trình build: {e}")