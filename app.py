# app.py
import gradio as gr
import logging
import time

# Import các module đã tạo
from config import (
    DATABASE_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    LLM_MODEL_NAME
)
from vector_store_loader import VectorStoreLoader
from llm_connector import LLMConnector
from legal_agent import SimpleLegalAgent
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. Khởi tạo toàn bộ hệ thống RAG và Agent ---
def initialize_system():
    """Khởi tạo và trả về các đối tượng cần thiết cho RAG."""
    print("🤖" + "=" * 60)
    print("        KHỞI TẠO HỆ THỐNG RAG CHO GIAO DIỆN WEB")
    print("=" * 65)

    # Tải lại Vector Database
    vector_store_loader = VectorStoreLoader(
        db_directory=DATABASE_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    vectordb = vector_store_loader.load()
    if not vectordb:
        raise Exception("Không thể tải Vector Database.")

    # Khởi tạo và kết nối LLM
    llm_connector = LLMConnector(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME
    )
    llm = llm_connector.connect()
    if not llm:
        raise Exception("Không thể kết nối LLM.")

    # --- Xây Dựng Lõi RAG (Retriever & Chain) ---
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # Khởi tạo Agent
    legal_agent = SimpleLegalAgent(
        retriever=retriever,
        llm=llm,
        rag_chain=rag_chain
    )

    logger.info("✅ Tất cả thành phần đã sẵn sàng!")
    return legal_agent


# --- Khởi tạo hệ thống một lần khi app được load ---
try:
    legal_agent = initialize_system()
except Exception as e:
    legal_agent = None
    logger.error(f"❌ Lỗi nghiêm trọng khi khởi tạo hệ thống: {e}")


# --- 2. Định nghĩa hàm xử lý cho Gradio ---
def chat_with_agent(question, history):
    if not legal_agent:
        return "Xin lỗi, hệ thống đang gặp lỗi. Vui lòng thử lại sau.", history

    start_time = time.time()
    try:
        # Sử dụng phương thức ask của Agent để xử lý câu hỏi
        answer = legal_agent.ask(question)
        end_time = time.time()

        # In kết quả xử lý
        logger.info(f"Thời gian xử lý: {end_time - start_time:.2f} giây")

        # Cập nhật lịch sử chat cho Gradio
        history.append((question, answer))
        return "", history

    except Exception as e:
        logger.error(f"Lỗi khi xử lý câu hỏi từ người dùng: {e}")
        error_message = f"Xin lỗi, có lỗi xảy ra trong quá trình xử lý: {e}"
        history.append((question, error_message))
        return "", history


# --- 3. Tạo giao diện Gradio ---
with gr.Blocks(title="Trợ Lý AI Pháp Lý (RAG + Agent)") as demo:
    gr.Markdown(
        """
        # 🤖 Trợ Lý AI Pháp Lý
        Chào mừng bạn đến với hệ thống hỏi đáp thông minh về Luật Bảo vệ dữ liệu cá nhân.
        Hãy nhập câu hỏi của bạn vào khung chat dưới đây!
        """
    )

    # Hiển thị lịch sử chat
    chatbot = gr.Chatbot(height=500)

    # Khung nhập liệu
    msg = gr.Textbox(
        label="Câu hỏi của bạn:",
        placeholder="Ví dụ: Dữ liệu cá nhân nhạy cảm là gì?"
    )

    # Nút gửi và nút xóa
    with gr.Row():
        clear_btn = gr.ClearButton([msg, chatbot], value="Xóa")
        submit_btn = gr.Button("Gửi", variant="primary")

    # Xử lý sự kiện khi người dùng gửi câu hỏi
    submit_btn.click(
        chat_with_agent,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    )

    msg.submit(
        chat_with_agent,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    )

    gr.Examples(
        examples=[
            "Dữ liệu dùng chung là gì?",
            "Sự khác nhau giữa dữ liệu dùng chung và dữ liệu dùng riêng?",
            "Cơ sở dữ liệu quốc gia được lưu trữ ở đâu?"
        ],
        inputs=msg
    )

if __name__ == "__main__":
    demo.launch(share=True)  # share=True để tạo link public cho Hugging Face Spaces