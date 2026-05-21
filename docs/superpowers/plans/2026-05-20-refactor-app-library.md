# App Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `App.py` to support a "My Library" tab for PDF management alongside an "AI Chat" tab using `gr.Tabs()`.

**Architecture:**
- `App.py` will use `gr.Blocks()` wrapping `gr.Tabs()`.
- "My Library" tab will use `gr.Dataframe` to list stored PDFs and `gr.File` for uploading.
- `StorageManager` will persist document metadata and coordinate file movement to `./data/pdfs/`.
- Uploading a file will trigger `index_pdf` and update the library table.

**Tech Stack:** Python, Gradio, LangChain, ChromaDB.

---

### Task 1: Update Storage Manager Integration

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Import and Initialize StorageManager**

```python
from storage_manager import StorageManager

storage = StorageManager()
```

- [ ] **Step 2: Implement PDF Upload Handler**

```python
def upload_file(file):
    if file is None:
        return "No file uploaded", get_library_data()
    
    # Copy file to data/pdfs
    filename = os.path.basename(file.name)
    destination = os.path.join(storage.pdfs_dir, filename)
    import shutil
    shutil.copy2(file.name, destination)
    
    # Index
    collection_id = filename.replace(".pdf", "").replace(" ", "_")
    index_pdf(destination, collection_id)
    storage.add_doc(filename, collection_id)
    
    return f"Uploaded {filename}", get_library_data()

def get_library_data():
    import pandas as pd
    docs = storage.get_all_docs()
    return pd.DataFrame(docs) if docs else pd.DataFrame(columns=["name", "id", "date"])
```

### Task 2: Refactor Gradio UI to use Tabs

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Reconstruct App structure**

```python
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title="RAG PDF Chatbot") as demo:
    gr.Markdown("# RAG PDF Chatbot")
    
    with gr.Tabs():
        with gr.TabItem("My Library"):
            file_upload = gr.File(label="Upload New PDF", file_types=[".pdf"])
            lib_df = gr.Dataframe(value=get_library_data(), label="Documents in Library")
            upload_status = gr.Markdown()
            file_upload.upload(upload_file, [file_upload], [upload_status, lib_df])
            
        with gr.TabItem("AI Chat"):
            # Original chat UI here...
```

- [ ] **Step 2: Cleanup existing global state variables and update UI links**

### Task 3: Verification

- [ ] **Step 1: Launch and Verify**

Run: `python App.py`
Expected: Gradio app opens with two tabs, library tab lists existing docs (if any), file upload adds docs.
