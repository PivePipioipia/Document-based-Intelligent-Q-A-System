# text_processor.py
import re
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# Cấu hình logging
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Lớp xử lý văn bản đầu vào, bao gồm đọc, làm sạch và các chiến lược phân đoạn thông minh.
    """
    def __init__(self):
        self.supported_formats = ['.txt', '.md', '.json']

    def read_file(self, file_path: str) -> str:
        """Đọc file văn bản với xử lý lỗi."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"✅ Đã đọc thành công {len(content)} ký tự từ {file_path.name}")
            return content
        except Exception as e:
            logger.error(f"❌ Lỗi khi đọc file: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """Làm sạch văn bản, loại bỏ khoảng trắng và các dòng trống thừa."""
        if not text or not text.strip():
            return ""

        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = '\n'.join(lines)

        logger.info(f"🧹 Văn bản sau khi làm sạch: {len(cleaned_text)} ký tự")
        return cleaned_text

    def _split_by_sentence(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        Chiến lược chunking cơ bản: chia theo câu và ghép lại.
        Đây là phương thức nội bộ (private method).
        """
        logger.info(f"Áp dụng chiến lược chunking theo câu (size={chunk_size}, overlap={overlap})...")
        if not text: return []

        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

        docs = text_splitter.split_text(text)

        chunks = []
        for i, doc_content in enumerate(docs):
            chunks.append({
                "id": f"chunk_sent_{datetime.now().timestamp()}_{i}",
                "content": doc_content,
                "length": len(doc_content)
            })

        logger.info(f"📝 Đã tạo {len(chunks)} chunks theo câu.")
        return chunks

    def _split_by_law_article(self, text: str, max_chars_per_chunk: int) -> List[Dict[str, Any]]:
        """
        Chiến lược chunking thông minh: chia theo cấu trúc Chương, Điều của văn bản luật.
        Đây là phương thức nội bộ (private method).
        """
        logger.info("Áp dụng chiến lược chunking thông minh theo Điều luật...")

        raw_parts = re.split(r'(Chương [IVXLCDM\d]+.*?)\n|(Điều \d+\..*?)\n', text)
        parts = [p.strip() for p in raw_parts if p and p.strip()]

        chunks = []
        current_heading = ""
        current_content = ""
        chunk_id_counter = 0

        for part in parts:
            is_heading = part.startswith("Chương") or part.startswith("Điều")

            if is_heading and current_content:
                full_content = (current_heading + "\n" + current_content).strip()
                chunks.append({
                    "id": f"chunk_law_{datetime.now().timestamp()}_{chunk_id_counter}",
                    "content": full_content,
                    "length": len(full_content),
                    "metadata": {"heading": current_heading}
                })
                chunk_id_counter += 1

                current_heading = part
                current_content = ""
            elif is_heading:
                current_heading = part
            else:
                current_content += "\n" + part

        if current_content:
            full_content = (current_heading + "\n" + current_content).strip()
            chunks.append({
                "id": f"chunk_law_{datetime.now().timestamp()}_{chunk_id_counter}",
                "content": full_content,
                "length": len(full_content),
                "metadata": {"heading": current_heading}
            })

        final_chunks = []
        for chunk in chunks:
            if chunk['length'] > max_chars_per_chunk:
                logger.warning(f"⚠️ Chunk '{chunk['metadata']['heading']}' quá dài ({chunk['length']} ký tự). Sẽ chia nhỏ hơn.")
                smaller_chunks_data = self._split_by_sentence(chunk['content'], max_chars_per_chunk, overlap=int(max_chars_per_chunk*0.1))
                final_chunks.extend(smaller_chunks_data)
            else:
                final_chunks.append(chunk)

        logger.info(f"📝 Đã tạo {len(final_chunks)} chunks dựa trên cấu trúc Điều/Chương.")
        return final_chunks

    def split_into_chunks(self, text: str, chunk_size: int = 1500, overlap: int = 150, strategy: str = "law_article") -> List[Dict[str, Any]]:
        """
        Phương thức chính để phân đoạn văn bản, có thể chọn chiến lược.
        """
        if strategy == "law_article":
            return self._split_by_law_article(text, max_chars_per_chunk=chunk_size)
        elif strategy == "sentence":
            return self._split_by_sentence(text, chunk_size, overlap)
        else:
            logger.error(f"❌ Chiến lược chunking không hợp lệ: {strategy}. Sử dụng 'sentence' làm mặc định.")
            return self._split_by_sentence(text, chunk_size, overlap)