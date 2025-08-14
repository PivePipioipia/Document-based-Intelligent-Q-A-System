# llm_connector.py
import logging
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

class LLMConnector:
    def __init__(self, groq_api_key: str, model_name: str):
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.llm = None

    def connect(self):
        """Kết nối và khởi tạo LLM thông qua Groq API"""
        try:
            logger.info(f"🔄 Đang kết nối tới Large Language Model (LLM): {self.model_name} qua Groq API...")
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name
            )

            # Thử gửi một câu lệnh đơn giản để kiểm tra kết nối
            self.llm.invoke("Xin chào!")

            logger.info(f"✅ Kết nối tới LLM '{self.model_name}' thành công!")
            return self.llm

        except Exception as e:
            logger.error(f"❌ Lỗi nghiêm trọng khi kết nối tới LLM: {e}")
            logger.error("Hãy đảm bảo rằng GROQ_API_KEY là chính xác và có kết nối internet.")
            return None