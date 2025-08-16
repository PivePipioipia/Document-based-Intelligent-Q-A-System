import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SimpleLegalAgent:
    """
    Phiên bản đơn giản hóa của Legal Agent với xử lý lỗi tốt hơn
    """
    def __init__(self, retriever, llm, rag_chain):
        self.retriever = retriever
        self.llm = llm
        self.rag_chain = rag_chain
        self.conversation_history = []  

        logger.info("Simple Legal Agent đã được khởi tạo thành công!")

    def search_documents(self, query: str) -> str:
        """Tìm kiếm cơ bản trong tài liệu"""
        try:
            logger.info(f"Tìm kiếm: {query}")
            result = self.rag_chain.invoke(query)

            # Thêm source citation
            docs = self.retriever.invoke(query)
            sources = []
            for i, doc in enumerate(docs[:2], 1):
                content = doc.page_content[:100]
                if "Điều" in content:
                    match = re.search(r'Điều \d+', content)
                    if match:
                        sources.append(match.group())

            if sources:
                result += f"\n\nNguồn: {', '.join(sources)}"

            return result
        except Exception as e:
            return f"Lỗi khi tìm kiếm: {str(e)}"

    def analyze_question_and_respond(self, question: str) -> str:
        """Phân tích câu hỏi và đưa ra phản hồi thông minh"""
        try:
            logger.info(f"Phân tích câu hỏi: {question}")

            # Phân loại loại câu hỏi
            question_lower = question.lower()

            if any(word in question_lower for word in ["so sánh", "khác biệt", "giống", "khác nhau"]):
                return self._handle_comparison_question(question)
            elif any(word in question_lower for word in ["định nghĩa", "là gì", "có nghĩa", "khái niệm"]):
                return self._handle_definition_question(question)
            elif any(word in question_lower for word in ["vi phạm", "phạt", "hậu quả", "trách nhiệm"]):
                return self._handle_compliance_question(question)
            elif any(word in question_lower for word in ["điều", "khoản", "quy định"]):
                return self._handle_article_question(question)
            else:
                return self._handle_general_question(question)

        except Exception as e:
            logger.error(f"Lỗi khi phân tích câu hỏi: {e}")
            return self.search_documents(question)  

    def _handle_definition_question(self, question: str) -> str:
        """Xử lý câu hỏi về định nghĩa"""
        logger.info("Xử lý câu hỏi định nghĩa")
        result = self.search_documents(question)

        # Tìm thêm thông tin liên quan
        if "định nghĩa" not in question.lower():
            extended_query = f"định nghĩa {question}"
            additional_info = self.search_documents(extended_query)
            if "không tìm thấy" not in additional_info.lower():
                result += f"\n\n📖 THÔNG TIN Bổ SUNG:\n{additional_info}"

        return result

    def _handle_comparison_question(self, question: str) -> str:
        """Xử lý câu hỏi so sánh"""
        logger.info("Xử lý câu hỏi so sánh")

        # Tìm kiếm thông tin chung trước
        result = self.search_documents(question)

        # Tìm từng khái niệm riêng lẻ
        words = question.split()
        concepts = [word for word in words if len(word) > 3 and word not in ['giữa', 'với', 'và', 'của', 'trong']]

        additional_info = ""
        for concept in concepts[:2]:  
            concept_query = f"định nghĩa {concept}"
            concept_info = self.search_documents(concept_query)
            if "không tìm thấy" not in concept_info.lower():
                additional_info += f"\n\n VỀ '{concept.upper()}':\n{concept_info}"

        if additional_info:
            result += additional_info

        return result

    def _handle_compliance_question(self, question: str) -> str:
        """Xử lý câu hỏi về tuân thủ"""
        logger.info(" Xử lý câu hỏi tuân thủ")

        # Tìm quy định
        result = self.search_documents(question)

        # Tìm hậu quả vi phạm
        violation_query = f"hình phạt vi phạm {question}"
        violation_info = self.search_documents(violation_query)
        if "không tìm thấy" not in violation_info.lower():
            result += f"\n\n HẬU QUẢ VI PHẠM:\n{violation_info}"

        return result

    def _handle_article_question(self, question: str) -> str:
        """Xử lý câu hỏi về điều khoản cụ thể"""
        logger.info(" Xử lý câu hỏi về điều khoản")

        result = self.search_documents(question)

        # Tìm điều khoản liên quan
        article_match = re.search(r'Điều \d+', question)
        if article_match:
            article = article_match.group()
            related_query = f"điều khoản liên quan {article}"
            related_info = self.search_documents(related_query)
            if "không tìm thấy" not in related_info.lower():
                result += f"\n\n ĐIỀU KHOẢN LIÊN QUAN:\n{related_info}"

        return result

    def _handle_general_question(self, question: str) -> str:
        """Xử lý câu hỏi chung"""
        logger.info(" Xử lý câu hỏi chung")
        return self.search_documents(question)

    def ask(self, question: str) -> str:
        """Phương thức chính để hỏi Agent"""
        try:
            print(f"\nAgent đang xử lý câu hỏi: '{question}'")
            print("Đang phân tích và tìm kiếm thông tin...")
            print("-" * 60)

            
            self.conversation_history.append({"role": "user", "content": question})

            
            answer = self.analyze_question_and_respond(question)

           
            self.conversation_history.append({"role": "assistant", "content": answer})

            print("\n" + "="*60)
            print("KẾT QUẢ TƯ VẤN PHÁP LÝ")
            print("="*60)
            print(answer)
            print("="*60)

            return answer

        except Exception as e:
            error_msg = f"Xin lỗi, có lỗi xảy ra: {str(e)}. Đang thử phương thức dự phòng..."
            print(f"{error_msg}")

            # Fallback: Sử dụng RAG chain trực tiếp
            try:
                fallback_answer = self.search_documents(question)
                print(f"\nKết quả dự phòng:\n{fallback_answer}")
                return fallback_answer
            except Exception as fallback_error:
                final_error = f"Không thể xử lý câu hỏi: {str(fallback_error)}"
                print(f"{final_error}")
                return final_error

    def ask_multiple_followup(self, main_question: str, followup_questions: List[str]) -> Dict[str, str]:
        """Hỏi một câu chính và nhiều câu hỏi phụ"""
        results = {}

        print(f"\nCÂU HỎI CHÍNH: {main_question}")
        results["main"] = self.ask(main_question)

        for i, followup in enumerate(followup_questions, 1):
            print(f"\nCÂU HỎI PHỤ {i}: {followup}")
            results[f"followup_{i}"] = self.ask(followup)

        return results

    def show_conversation_history(self):
        """Hiển thị lịch sử hội thoại"""
        if not self.conversation_history:
            print("Chưa có lịch sử hội thoại.")
            return

        print("\n LỊCH SỬ HỘI THOẠI:")
        print("-" * 50)
        for entry in self.conversation_history:
            if entry["role"] == "user":
                print(f"Người dùng: {entry['content']}")
            else:
                print(f"Agent: {entry['content'][:200]}...")  # hiển thị 200 ký tự đầu
            print("-" * 30)

    def clear_memory(self):
        """Xóa lịch sử hội thoại"""
        self.conversation_history = []
        print("Đã xóa lịch sử hội thoại.")
