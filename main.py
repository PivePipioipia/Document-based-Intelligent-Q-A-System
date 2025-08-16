import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from legal_agent import SimpleLegalAgent  

from config import DATABASE_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, GROQ_API_KEY, LLM_MODEL_NAME
from vector_store_loader import VectorStoreLoader
from llm_connector import LLMConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def format_docs(docs):
    """Hàm hỗ trợ để định dạng các tài liệu truy xuất thành một chuỗi duy nhất."""
    return "\n\n".join(doc.page_content for doc in docs)

PROMPT_TEMPLATE = """
Bạn là một trợ lý AI pháp lý, chuyên trả lời các câu hỏi dựa trên nội dung của văn bản Luật được cung cấp.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác và chỉ dựa vào thông tin có trong phần "NGỮ CẢNH" dưới đây.

**NGỮ CẢNH:**
{context}

**DỰA VÀO NGỮ CẢNH TRÊN, HÃY TRẢ LỜI CÂU HỎI SAU:**
**Câu hỏi:** {question}

**QUY TẮC TRẢ LỜI:**
- Trả lời thẳng vào vấn đề, không thêm lời chào hay các câu nói không liên quan.
- Nếu câu trả lời có trong ngữ cảnh, hãy trích dẫn lại thông tin một cách ngắn gọn.
- **Nếu câu trả lời không thể được tìm thấy trong ngữ cảnh, hãy trả lời chính xác là: "Tôi không tìm thấy thông tin về điều này trong tài liệu được cung cấp."**
- Không được suy diễn, phỏng đoán hay sử dụng kiến thức bên ngoài ngữ cảnh.
- Luôn trả lời bằng tiếng Việt.
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

if __name__ == "__main__":
    print("🤖" + "=" * 60)
    print("        TRỢ LÝ AI PHÁP LÝ THÔNG MINH - HỆ THỐNG RAG")
    print("=" * 65)
    logger.info("Bắt đầu khởi tạo hệ thống RAG...")

    # --- Bước 2: Tải lại Kho Tri Thức và Khởi tạo Mô Hình ---
    logger.info("Tải lại kho tri thức và khởi tạo mô hình...")

    vector_store_loader = VectorStoreLoader(
        db_directory=DATABASE_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    vectordb = vector_store_loader.load()
    if not vectordb:
        exit()

    llm_connector = LLMConnector(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME
    )
    llm = llm_connector.connect()
    if not llm:
        exit()

    print("-" * 60)
    logger.info("Hoàn thành. Bắt đầu xây dựng RAG Chain.")

    try:
        base_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        logger.info(" Đã khởi tạo Retriever cơ bản thành công")

        logger.info(" Đang khởi tạo Multi-Query Retriever...")
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        logger.info(" Đã khởi tạo Multi-Query Retriever thành công")
    except Exception as e:
        logger.error(f" Lỗi khi khởi tạo Retriever: {e}")
        exit()

    try:
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        logger.info(" Đã lắp ráp RAG Chain hoàn chỉnh!")
    except Exception as e:
        logger.error(f" Lỗi khi lắp ráp RAG Chain: {e}")
        exit()

    try:
        print("-" * 60)
        logger.info(" Đang khởi tạo Simple Legal Agent...")

        legal_agent = SimpleLegalAgent(
            retriever=retriever,
            llm=llm,
            rag_chain=rag_chain
        )

        print("\n SIMPLE LEGAL AGENT ĐÃ SẴN SÀNG!")
        print(" Gõ 'exit' để thoát.")
        print(" Gõ 'history' để xem lịch sử.")
        print(" Gõ 'clear' để xóa lịch sử.")

        while True:
            question = input("\nBạn hỏi gì?")
            if question.lower() == 'exit':
                break
            elif question.lower() == 'history':
                legal_agent.show_conversation_history()
            elif question.lower() == 'clear':
                legal_agent.clear_memory()
            elif question.strip() == "":
                print("Vui lòng nhập câu hỏi.")
            else:
                legal_agent.ask(question)

    except Exception as e:
        logger.error(f" Lỗi khi khởi tạo và chạy Agent: {e}")
        print(f" Hệ thống đã gặp lỗi: {e}")
