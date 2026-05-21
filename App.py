import os
import shutil
import warnings
import gradio as gr
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from functools import lru_cache
from dotenv import load_dotenv

from storage_manager import StorageManager

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.prompts import PromptTemplate
from langchain_core.messages import ChatMessageHistory, HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

load_dotenv()
warnings.filterwarnings("ignore")

storage = StorageManager()

# ── Config ────────────────────────────────────────────────────────────────────
LLM_BACKEND  = os.getenv("LLM_BACKEND",  "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./data/vector_db")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_library_data():
    docs = storage.get_all_docs()
    return pd.DataFrame(docs) if docs else pd.DataFrame(columns=["name", "id", "date"])

def upload_file(file):
    if file is None:
        return "No file uploaded", get_library_data(), gr.Dropdown(choices=get_doc_choices())
    
    filename = os.path.basename(file.name)
    destination = os.path.join(storage.pdfs_dir, filename)
    shutil.copy2(file.name, destination)
    
    collection_id = filename.replace(".pdf", "").replace(" ", "_")
    index_pdf(destination)
    storage.add_doc(filename, collection_id)
    
    return f"Uploaded {filename}", get_library_data(), gr.Dropdown(choices=get_doc_choices())

def get_doc_choices():
    docs = storage.get_all_docs()
    return [(d['name'], d['id']) for d in docs]

# ── LLM & Embeddings ──────────────────────────────────────────────────────────
@lru_cache()
def get_llm():
    if LLM_BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)

@lru_cache()
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vector_store(collection_name):
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=VECTOR_DB_DIR
    )

# Helper to check if text needs OCR
def needs_ocr(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        if page.extract_text().strip():
            return False
    return True

# Helper to perform OCR
def perform_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text

def index_pdf(pdf_path):
    vectordb = get_vector_store("library")
    existing = vectordb.get()
    if existing and existing.get("ids"):
        return vectordb

    if needs_ocr(pdf_path):
        print("Scanned PDF detected, performing OCR...")
        text = perform_ocr(pdf_path)
        # Create a Document object from text
        from langchain.docstore.document import Document
        docs = [Document(page_content=text)]
    else:
        docs = PyPDFLoader(pdf_path).load()
        
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    
    return Chroma.from_documents(
        chunks, 
        get_embeddings(), 
        collection_name="library", 
        persist_directory=VECTOR_DB_DIR
    )

def build_rag_chain():
    llm = get_llm()
    retriever = get_vector_store("library").as_retriever(search_kwargs={"k": 4})

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

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def chat_respond(message, history):
    conversational_rag_chain = build_rag_chain()

    with_message_history = RunnableWithMessageHistory(
        conversational_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Convert gradio history to LangChain format
    langchain_history = []
    for human, ai in history:
        langchain_history.append(HumanMessage(content=human))
        langchain_history.append(AIMessage(content=ai))
    
    config = {"configurable": {"session_id": "gradio_chat_session"}}
    
    full_response = ""
    for chunk in with_message_history.stream({"input": message, "chat_history": langchain_history}, config=config):
        if "answer" in chunk:
            full_response += chunk["answer"]
        history.append((message, full_response))
        yield "", history
        history.pop()
    
    history.append((message, full_response))
    return "", history

# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title="RAG PDF Chatbot") as demo:

    gr.Markdown("# RAG PDF Chatbot")
    
    with gr.Tabs():
        with gr.TabItem("My Library"):
            file_upload = gr.File(label="Upload New PDF", file_types=[".pdf"])
            lib_df = gr.Dataframe(value=get_library_data(), label="Documents in Library")
            upload_status = gr.Markdown()
            
            # Re-defined in code block to avoid name collision
            doc_dropdown = gr.Dropdown(label="Select Document", choices=get_doc_choices())
            file_upload.upload(upload_file, [file_upload], [upload_status, lib_df, doc_dropdown])

        with gr.TabItem("AI Chat"):
            chatbot = gr.Chatbot(label="Conversation")
            msg_input = gr.Textbox(placeholder="Ask anything...")
            clear = gr.Button("Clear Chat")
            
            msg_input.submit(chat_respond, [msg_input, chatbot], [msg_input, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_port=7861)
