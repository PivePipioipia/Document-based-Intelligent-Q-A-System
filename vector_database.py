# vector_database.py
import logging
from typing import List, Dict, Any
from pathlib import Path
import chromadb

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Lưu trữ và quản lý vector database bằng ChromaDB"""
    def __init__(self, db_path: str, collection_name: str = "documents"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.db_path.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info(f"✅ Đã kết nối/tạo thành công collection '{self.collection_name}' tại '{self.db_path}'")

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], source_file: str):
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            logger.error("❌ Số lượng chunks và embeddings không khớp.")
            return False
        try:
            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['content'] for chunk in chunks]
            metadatas = [{'source_file': source_file, 'length': chunk['length']} for chunk in chunks]
            self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            logger.info(f"✅ Đã thêm {len(chunks)} documents vào vector database.")
            return True
        except Exception as e:
            logger.error(f"❌ Lỗi khi thêm documents: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        try:
            return {
                'database_path': str(self.db_path),
                'collection_name': self.collection_name,
                'total_documents': self.collection.count()
            }
        except Exception as e:
            logger.error(f"❌ Lỗi khi lấy thông tin database: {e}")
            return {}