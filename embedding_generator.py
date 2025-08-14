# embedding_generator.py
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Tạo vector embeddings từ văn bản"""
    def __init__(self, model_name: str):
        try:
            logger.info(f"🔄 Đang tải embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.vector_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Đã tải model '{model_name}' thành công. Dimension: {self.vector_dimension}")
        except Exception as e:
            logger.warning(f"⚠️ Không thể tải model {model_name}. Lỗi: {e}. Đang chuyển sang model dự phòng.")
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(fallback_model)
            self.model_name = fallback_model
            self.vector_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Đã tải model dự phòng: {fallback_model}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        try:
            logger.info(f"🔄 Đang tạo embeddings cho {len(texts)} đoạn văn...")
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
            embeddings_list = embeddings.tolist()
            logger.info(f"✅ Đã tạo {len(embeddings_list)} embeddings")
            return embeddings_list
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo embeddings: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'vector_dimension': self.vector_dimension
        }