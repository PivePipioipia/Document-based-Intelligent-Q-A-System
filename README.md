# Trợ Lý AI Pháp Lý (RAG + Agent)

## Giới Thiệu
Đây là một hệ thống hỏi đáp thông minh được xây dựng để cung cấp thông tin pháp lý dựa trên nội dung của văn bản **Luật Bảo vệ Dữ liệu Cá nhân** của Việt Nam. Hệ thống sử dụng kiến trúc **Retrieval-Augmented Generation (RAG)** kết hợp với một **Agent** thông minh để tìm kiếm, phân tích và trả lời các câu hỏi của người dùng một cách chính xác và hiệu quả.

## Các Thành Phần Chính
-   **Vector Database (ChromaDB):** Lưu trữ các đoạn văn bản từ luật đã được mã hóa thành vector (embedding).
-   **Embedding Model:** Sử dụng `sentence-transformers/multi-qa-mpnet-base-dot-v1` để chuyển đổi văn bản thành các vector số học.
-   **Large Language Model (LLM):** Sử dụng `mixtral-8x7b-32768` từ Groq để tạo câu trả lời dựa trên ngữ cảnh đã được truy xuất.
-   **RAG Chain:** Một chuỗi xử lý kết hợp việc truy xuất thông tin từ Vector DB và tạo câu trả lời bằng LLM.
-   **Legal Agent:** Một bộ não thông minh hơn RAG thông thường, có khả năng phân tích loại câu hỏi (định nghĩa, so sánh, vi phạm,...) để đưa ra câu trả lời chi tiết và liên quan hơn.
-   **Giao Diện Người Dùng (Gradio):** Một giao diện web đơn giản nhưng mạnh mẽ, cho phép người dùng tương tác trực tiếp với Agent.

## Demo
Bạn có thể trải nghiệm trực tiếp Trợ Lý AI Pháp Lý tại đây:
**[https://huggingface.co/spaces/nguyen-hong-yen/my-legal-agent]**

## Cài Đặt và Chạy Dự Án Cục Bộ



### 1. Cài đặt môi trường
Tạo và kích hoạt virtual environment
python -m venv .venv

Trên Windows
.venv\Scripts\activate

Trên macOS/Linux
source .venv/bin/activate

### 2.Cài đặt các Thư viện
pip install -r requirements.txt

### 3. Cấu hình API Key
Tạo một file có tên .env trong thư mục gốc và thêm khóa API của Groq.
GROQ_API_KEY="your-groq-api-key"

### 4. Xây dựng Cơ sở Tri thức
Chạy file build_kb.py để xử lý văn bản luật và tạo Vector Database.
python build_kb.py
### 5. Chạy local
python main.py

