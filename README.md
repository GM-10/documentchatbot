# 🤖 AI Knowledge Assistant: Professional RAG PDF Chatbot

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system that transforms static PDF libraries into an interactive, verifiable knowledge base. This project demonstrates a complete AI pipeline from raw document ingestion (including OCR) to high-precision hybrid retrieval and hallucination-resistant generation.

---

## 🚀 Key Technical Achievements

### 1. Hybrid Search Architecture
Unlike basic RAG systems that rely solely on semantic vectors, this system implements a **Hybrid Retrieval** strategy using an `EnsembleRetriever`:
- **Dense Retrieval (Semantic)**: Uses HuggingFace embeddings to capture conceptual meaning and context.
- **Sparse Retrieval (BM25)**: Uses keyword-based matching to ensure exact terms, technical IDs, and rare names are never missed.
- **RRF (Reciprocal Rank Fusion)**: Intelligently merges both results to provide the most relevant context to the LLM.

### 2. Selective OCR Pipeline
To handle real-world documents, the system includes a robust ingestion pipeline:
- **Intelligent Detection**: Automatically detects "scanned" pages (empty or low-text PDF pages).
- **OCR Integration**: Uses `pytesseract` and `pdf2image` to perform Optical Character Recognition on scanned content, ensuring no data is lost during indexing.

### 3. Hallucination Control & Citation Verification
To ensure trust and accuracy, I implemented a custom **Citation Verification Engine**:
- **Strict Formatting**: The LLM is constrained to cite sources as `[source: filename.pdf | page: X]`.
- **Backend Validation**: Every response is post-processed to verify that the cited page and file actually exist in the retrieved context.
- **Transparency**: The UI explicitly notifies the user if citations are verified or if potential hallucinations were detected.

### 4. Enterprise-Grade UX
- **Side-by-Side PDF Viewer**: Integrated a real-time document preview so users can visually verify AI claims.
- **Conversational Memory**: Implemented a history-aware retriever that can rephrase user queries based on the chat context.
- **Multi-Doc Synthesis**: Capability to chat across the entire library or target a specific document.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM** | Ollama (Llama 3.2) / OpenAI (GPT-4o) |
| **Orchestration** | LangChain |
| **Vector Store** | ChromaDB |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **OCR** | Tesseract OCR, Poppler |
| **Frontend** | Gradio |
| **Deployment** | Docker, Docker Compose |

---

## 📦 Installation & Setup

### Prerequisites
- Docker installed (Recommended)
- Ollama installed and running (if using local LLM)

### Quick Start (Docker)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-pdf-chatbot.git
cd rag-pdf-chatbot

# Setup environment variables
cp .env.example .env

# Launch with Docker Compose
docker-compose up --build
```
Access the app at `http://localhost:7861`

---

## 🏗️ Project Structure
```
rag-pdf-chatbot/
├── App.py              # Main RAG pipeline & Gradio UI
├── config.py           # Centralized environment & app configuration
├── storage_manager.py  # Metadata and file system management
├── Dockerfile           # Containerization logic
├── docker-compose.yml  # Deployment orchestration
├── data/               # Persistent storage for PDFs and Vector DB
└── requirements.txt    # Python dependencies
```

---

## 🎓 Learning Outcomes
This project solved several critical RAG challenges:
- **The "Exact Match" Problem**: Solved by implementing Hybrid Search.
- **The "Scanned PDF" Problem**: Solved by Selective OCR.
- **The "Hallucination" Problem**: Solved by a custom Metadata Verification layer.
