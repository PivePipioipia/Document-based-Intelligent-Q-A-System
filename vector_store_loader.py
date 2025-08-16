import logging
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

class VectorStoreLoader:
    def __init__(self, db_directory: str, collection_name: str, embedding_model_name: str):
        self.db_directory = db_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_function = None
        self.vectordb = None

    def load(self):
        """Tải lại Vector Database đã lưu"""
        try:
            logger.info(" Đang khởi tạo embedding function...")
            self.embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

            logger.info(f" Đang tải lại Vector Database từ đường dẫn: {self.db_directory}")
            self.vectordb = Chroma(
                persist_directory=self.db_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name
            )

            logger.info(
                f" Tải lại Vector Database thành công. Số lượng documents: {self.vectordb._collection.count()}")
            return self.vectordb
        except Exception as e:
            logger.error(f" Lỗi nghiêm trọng khi tải lại Vector Database: {e}")
            logger.error("Kiểm tra bước trên đường dẫn tới DB")
            return None
