# Digital Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the RAG PDF chatbot into a persistent application with a multi-document library and enhanced two-tab UI.

**Architecture:** 
- **Persistence:** Use local ChromaDB directory and a `metadata.json` for document tracking.
- **UI:** Refactor `App.py` to use `gr.Tabs` for "Library" and "Chat" views.
- **State:** Use a global registry or metadata file to load/save document info across restarts.

**Tech Stack:** Python, Gradio, LangChain (Classic), ChromaDB, PyPDF.

---

### Task 1: Setup Storage Structure and Metadata Manager

**Files:**
- Create: `storage_manager.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing test for metadata management**
```python
import os
import json
import pytest
from storage_manager import StorageManager

def test_metadata_init():
    sm = StorageManager(base_dir="test_data")
    assert os.path.exists("test_data/metadata.json")
    assert sm.get_all_docs() == []
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_storage.py`

- [ ] **Step 3: Implement StorageManager**
```python
import os
import json
import shutil

class StorageManager:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.pdfs_dir = os.path.join(base_dir, "pdfs")
        self.db_dir = os.path.join(base_dir, "vector_db")
        self.metadata_path = os.path.join(base_dir, "metadata.json")
        
        os.makedirs(self.pdfs_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'w') as f:
                json.dump([], f)

    def get_all_docs(self):
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def add_doc(self, name, collection_id):
        docs = self.get_all_docs()
        docs.append({"name": name, "id": collection_id, "date": "2026-05-19"})
        with open(self.metadata_path, 'w') as f:
            json.dump(docs, f)

    def delete_doc(self, collection_id):
        docs = self.get_all_docs()
        new_docs = [d for d in docs if d['id'] != collection_id]
        with open(self.metadata_path, 'w') as f:
            json.dump(new_docs, f)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_storage.py`

---

### Task 2: Refactor RAG Pipeline for Persistence

**Files:**
- Modify: `App.py`
- Update: `get_embeddings` and `build_rag_chain`

- [ ] **Step 1: Update RAG logic to use persistent Chroma**
Modify `build_rag_chain` to accept a `collection_name` and use `persist_directory`.

```python
def get_vector_store(collection_name):
    from langchain_community.vectorstores import Chroma
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory="./data/vector_db"
    )

def index_pdf(pdf_path, collection_name):
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    vectordb = Chroma.from_documents(
        chunks, 
        get_embeddings(), 
        collection_name=collection_name, 
        persist_directory="./data/vector_db"
    )
    return vectordb
```

- [ ] **Step 2: Verify import and persistence**
Run: `python -c "from App import get_vector_store; print('Success')"`

---

### Task 3: Implement Library Tab UI

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Create Library Tab layout**
Use `gr.Tab("Library")` with a `gr.File` for upload and a `gr.Dataframe` or `gr.List` for existing docs.

```python
with gr.Tab("My Library"):
    upload_btn = gr.File(label="Add New PDF", file_types=[".pdf"])
    library_table = gr.Dataframe(headers=["Name", "ID", "Date"], value=[])
    refresh_btn = gr.Button("Refresh Library")
```

- [ ] **Step 2: Add upload handler**
Create a function that saves the file, indexes it, and updates metadata.

---

### Task 4: Implement Chat Tab UI with History

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Create Chat Tab layout**
Use `gr.Tab("Chat")` with `gr.Chatbot` and `gr.Dropdown` for document selection.

```python
with gr.Tab("AI Chat"):
    doc_select = gr.Dropdown(label="Select Document", choices=[])
    chatbot = gr.Chatbot(label="Conversation")
    msg_input = gr.Textbox(placeholder="Ask anything...")
    clear = gr.Button("Clear Chat")
```

- [ ] **Step 2: Implement Chat logic**
Update the `answer` function to work with `gr.Chatbot` (appending to history).

---

### Task 5: Integration and Cleanup

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Connect Library and Chat**
Ensure clicking "Refresh" updates the dropdown choices in the Chat tab.
Implement the delete functionality.

- [ ] **Step 2: Final Verification**
Run the app, upload two different PDFs, switch between them in chat, and ensure they persist after restart.
