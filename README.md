# RAG PDF Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask questions — answers are grounded in the document, not hallucinated.

Built with LangChain, ChromaDB, HuggingFace embeddings, and Gradio. Supports both **Ollama (free, local)** and **OpenAI** as LLM backends.

---

## Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader  ──►  RecursiveCharacterTextSplitter (chunk_size=1000)
                            │
                            ▼
              HuggingFace Embeddings (all-MiniLM-L6-v2)
                            │
                            ▼
                       ChromaDB (in-memory vector store)
                            │
              ┌─────────────┘
              │   User Query
              │       │
              ▼       ▼
        Similarity Search  ──►  Top-4 chunks
                                    │
                                    ▼
                            LLM (Ollama / OpenAI)
                                    │
                                    ▼
                               Final Answer
```

---

## Tech Stack

| Component     | Tool                                     |
|---------------|------------------------------------------|
| Framework     | LangChain                                |
| LLM           | Ollama `llama3.2` (local) / OpenAI GPT  |
| Embeddings    | HuggingFace `all-MiniLM-L6-v2` (free)   |
| Vector Store  | ChromaDB (in-memory)                     |
| PDF Loader    | PyPDF via LangChain                      |
| UI            | Gradio                                   |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-pdf-chatbot
cd rag-pdf-chatbot
pip install -r requirements.txt
```

### 2. Set up your LLM

**Option A — Ollama (free, recommended)**

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2
ollama serve
```

**Option B — OpenAI**

```bash
cp .env.example .env
# Open .env and set:
#   LLM_BACKEND=openai
#   OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## How It Works

1. **Load** — The PDF is loaded page by page using PyPDFLoader.
2. **Chunk** — Text is split into 1000-character chunks with 100-character overlap to preserve context at boundaries.
3. **Embed** — Each chunk is converted to a vector using a free HuggingFace sentence-transformer model.
4. **Store** — Vectors are stored in an in-memory ChromaDB collection.
5. **Retrieve** — At query time, the top 4 most semantically similar chunks are retrieved.
6. **Generate** — The LLM answers using only the retrieved chunks, keeping answers grounded in the document.

---

## Configuration

All settings are controlled via environment variables (see `.env.example`):

| Variable       | Default              | Description                        |
|----------------|----------------------|------------------------------------|
| `LLM_BACKEND`  | `ollama`             | `ollama` or `openai`               |
| `OLLAMA_MODEL` | `llama3.2`           | Any model available via Ollama     |
| `OPENAI_MODEL` | `gpt-3.5-turbo`      | OpenAI model name                  |
| `EMBED_MODEL`  | `all-MiniLM-L6-v2`   | HuggingFace embedding model        |

---

## Example Questions to Try

- *"What is the main topic of this document?"*
- *"Summarize the key findings in bullet points."*
- *"What methodology was used?"*
- *"What does the author conclude about X?"*

---

## Project Structure

```
rag-pdf-chatbot/
├── app.py              # Main application (RAG pipeline + Gradio UI)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
```

---

## License

MIT