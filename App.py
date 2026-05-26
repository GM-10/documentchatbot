import os
import shutil
import warnings
import gradio as gr
import pandas as pd
import pytesseract
import json
import re
from pdf2image import convert_from_path
from functools import lru_cache
import config

from storage_manager import StorageManager

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from pypdf import PdfReader

# ── Init ───────────────────────────────────────────────────────────────────────

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_library_data():
    docs = storage.get_all_docs()
    return pd.DataFrame(docs) if docs else pd.DataFrame(columns=["name", "id", "date"])

def upload_file(file, progress=gr.Progress()):
    if file is None:
        return None, "No file uploaded", get_library_data(), get_doc_choices()

    filename = os.path.basename(file.name)
    destination = os.path.join(storage.pdfs_dir, filename)

    progress(0, desc="Uploading file...")
    shutil.copy2(file.name, destination)

    collection_id = filename.replace(".pdf", "").replace(" ", "_")

    progress(0.3, desc="Indexing PDF (this may take a while)...")

    # Get page count before indexing
    try:
        reader = PdfReader(destination)
        page_count = len(reader.pages)
    except Exception:
        page_count = 0

    index_pdf(destination, collection_id, progress)

    storage.add_doc(filename, collection_id, page_count=page_count)

    progress(1.0, desc="Upload complete!")
    return None, f"Uploaded {filename}", get_library_data(), get_doc_choices()

def get_doc_choices():
    docs = storage.get_all_docs()
    choices = [("All Documents", "all")] + [(d['name'], d['id']) for d in docs]
    print(f"get_doc_choices returning: {choices}")
    return choices

# ── LLM & Embeddings ──────────────────────────────────────────────────────────
@lru_cache()
def get_llm():
    if config.LLM_BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.3)
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=config.OLLAMA_MODEL, temperature=0.3)

@lru_cache()
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vector_store(collection_name):
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=config.VECTOR_DB_DIR
    )

# Helper to process PDF with selective OCR
def process_pdf_with_ocr(pdf_path, collection_id):
    reader = PdfReader(pdf_path)
    # Convert PDF to images for OCR (one image per page)
    images = convert_from_path(pdf_path)

    documents = []
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        text = page.extract_text()

        # If text is empty or very short, it's likely a scanned page
        if not text or len(text.strip()) < 50:
            print(f"Page {page_num}: Scanned content detected. Performing OCR...")
            # Use the corresponding image for this page
            if i < len(images):
                text = pytesseract.image_to_string(images[i])
            else:
                text = ""
        else:
            print(f"Page {page_num}: Text extracted successfully.")

        from langchain.docstore.document import Document
        documents.append(Document(
            page_content=text,
            metadata={"source": collection_id, "page": page_num}
        ))

    return documents

def index_pdf(pdf_path, collection_id, progress=None):
    vectordb = get_vector_store("library")
    existing = vectordb.get()
    # Check if this specific document (collection_id) has already been indexed
    if existing and any(collection_id == doc_id for doc_id in existing.get("ids", [])):
        if progress:
            progress(0.5, desc="Document already indexed. Skipping.")
        print(f"Document {collection_id} already indexed. Skipping.")
        return vectordb

    if progress:
        progress(0.4, desc="Processing PDF pages (Selective OCR)...")

    docs = process_pdf_with_ocr(pdf_path, collection_id)

    if progress:
        progress(0.7, desc="Splitting text into chunks...")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents(docs)

    if progress:
        progress(0.8, desc="Adding to vector database...")
    return Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name="library",
        persist_directory=config.VECTOR_DB_DIR
    )

def build_rag_chain(selected_doc_id=None):
    llm = get_llm()

    # 1. Semantic Retriever (Chroma)
    search_kwargs = {"k": 4}
    if selected_doc_id and selected_doc_id != "all":
        search_kwargs["filter"] = {"source": selected_doc_id}

    semantic_retriever = get_vector_store("library").as_retriever(search_kwargs=search_kwargs)

    # 2. Keyword Retriever (BM25)
    # Retrieve all documents from the vector store to initialize BM25
    vectordb = get_vector_store("library")
    all_data = vectordb.get()

    # Convert raw Chroma data back into LangChain Document objects
    from langchain.docstore.document import Document
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    # Filter documents for BM25 if a specific doc is selected
    if selected_doc_id and selected_doc_id != "all":
        all_docs = [doc for doc in all_docs if doc.metadata.get("source") == selected_doc_id]

    if not all_docs:
        # Fallback to just semantic if no docs available for BM25
        retriever = semantic_retriever
    else:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 4
        # Hybrid search combining both (weights are equal by default)
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]
        )

    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_retriever_prompt)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}\n\nAlways cite sources as [source: filename.pdf | page: X]"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, rag_prompt)

    conversational_rag_chain = create_retrieval_chain(history_aware_retriever, stuff_documents_chain)

    return conversational_rag_chain

# ── Chat Handler ──────────────────────────────────────────────────────────────

HISTORY_FILE = os.path.join("data", "chat_history.json")
store = {}

def save_history():
    serialized_history = {}
    for session_id, history_obj in store.items():
        serialized_history[session_id] = [
            {"role": msg.type, "content": msg.content}
            for msg in history_obj.messages
        ]

    with open(HISTORY_FILE, "w") as f:
        json.dump(serialized_history, f)

def load_history():
    global store
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                for session_id, messages in data.items():
                    history = ChatMessageHistory()
                    for msg in messages:
                        if msg["role"] == "human":
                            history.add_user_message(msg["content"])
                        elif msg["role"] == "ai":
                            history.add_ai_message(msg["content"])
                    store[session_id] = history
        except Exception as e:
            print(f"Error loading history: {e}")

# Initial load of history
load_history()

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def verify_citations(answer, context_docs):
    """Verify that citations in the answer exist in the retrieved context."""
    if not context_docs:
        return "⚠️ No context documents were retrieved to verify citations."

    # Find all citations in the format [source: filename.pdf | page: X]
    citation_pattern = r'\[source:\s*(.*?)\s*\|\s*page:\s*(\d+)\]'
    citations = re.findall(citation_pattern, answer)

    if not citations:
        return "ℹ️ No citations found in the answer."

    valid_count = 0
    for source, page in citations:
        page = int(page)
        # Check if any doc in context matches this source and page
        if any(doc.metadata.get("source") == source and doc.metadata.get("page") == page for doc in context_docs):
            valid_count += 1

    if valid_count == len(citations):
        return "✅ All citations verified against source documents."
    elif valid_count > 0:
        return f"⚠️ {valid_count}/{len(citations)} citations verified. Some may be inaccurate."
    else:
        return "❌ None of the citations could be verified against the source documents."

def update_pdf_viewer(doc_id):
    if not doc_id or doc_id == "all":
        return None

    docs = storage.get_all_docs()
    doc = next((d for d in docs if d['id'] == doc_id), None)
    if not doc:
        return None

    pdf_path = os.path.join(storage.pdfs_dir, doc['name'])
    return pdf_path

def chat_respond(message, history, collection_id):

    print(f"chat_respond received collection_id: {collection_id}")
    conversational_rag_chain = build_rag_chain(selected_doc_id=collection_id)

    with_message_history = RunnableWithMessageHistory(
        conversational_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    config = {"configurable": {"session_id": "gradio_chat_session"}}

    full_response = ""
    context_docs = []

    # RunnableWithMessageHistory automatically handles history via session_id
    for chunk in with_message_history.stream({"input": message}, config=config):
        if "context" in chunk:
            context_docs = chunk["context"]
        if "answer" in chunk:
            full_response += chunk["answer"]
            # Update Gradio chatbot with current accumulated response
            history.append((message, full_response))
            yield "", history
            history.pop()

    # Post-processing: Verify citations
    verification_msg = verify_citations(full_response, context_docs)
    final_answer = f"{full_response}\n\n{verification_msg}"

    history.append((message, final_answer))
    save_history() # Persist history after the full response is generated
    return "", history

def delete_document(doc_id):
    if not doc_id:
        return "No document selected", get_library_data(), get_doc_choices()

    # Remove from metadata
    storage.delete_doc(doc_id)

    # Remove from Vector DB
    try:
        vectordb = get_vector_store("library")
        # Find documents matching the source
        results = vectordb.get(where={"source": doc_id})
        ids = results.get("ids", [])
        if ids:
            vectordb.delete(ids=ids)
            print(f"Deleted {len(ids)} chunks from vector DB for {doc_id}")
    except Exception as e:
        print(f"Error deleting from vector DB: {e}")

    print(f"Deleted document: {doc_id}")
    return f"Deleted document {doc_id}", get_library_data(), get_doc_choices()

# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="RAG PDF Chatbot") as demo:

    gr.Markdown("# 🤖 AI Knowledge Assistant")

    with gr.Row():
        # Sidebar: Library Management
        with gr.Column(scale=1):
            gr.Markdown("## 📚 My Library")
            file_upload = gr.File(label="Upload New PDF", file_types=[".pdf"])
            lib_df = gr.Dataframe(value=get_library_data(), label="Documents")
            upload_status = gr.Markdown()

            with gr.Row():
                delete_doc_dropdown = gr.Dropdown(label="Delete Doc", choices=get_doc_choices())
                delete_btn = gr.Button("Delete", variant="stop")

        # Main Area: Chat and Preview
        with gr.Column(scale=3):
            with gr.Row():
                doc_dropdown = gr.Dropdown(
                    label="Select Document for Chat",
                    choices=get_doc_choices(),
                    scale=4
                )
                clear = gr.Button("Clear Chat", scale=1)

            # PDF Viewer (using gr.File for built-in preview)
            pdf_viewer = gr.File(label="PDF Preview", interactive=False)

            chatbot = gr.Chatbot(label="Conversation", height=500)
            msg_input = gr.Textbox(placeholder="Ask anything...", show_label=False)

    # Define events after all components are declared
    file_upload.upload(upload_file, [file_upload], [file_upload, upload_status, lib_df, doc_dropdown])

    delete_btn.click(
        delete_document,
        [delete_doc_dropdown],
        [upload_status, lib_df, doc_dropdown, delete_doc_dropdown]
    )

    doc_dropdown.change(update_pdf_viewer, [doc_dropdown], [pdf_viewer])

    msg_input.submit(chat_respond, [msg_input, chatbot, doc_dropdown], [msg_input, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_port=7861, allowed_paths=[os.path.abspath(storage.pdfs_dir)])
