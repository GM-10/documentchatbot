# Design Spec: RAG PDF Digital Library

## 1. Overview
Enhance the current single-session RAG PDF chatbot into a persistent "Digital Library" application. Users will be able to maintain a collection of PDFs, select them for chat, and preserve their indexed data across restarts.

## 2. Requirements
- **Persistence:** All uploaded PDFs and their vector embeddings must be saved locally.
- **Library Management:** Users can upload, list, and delete documents from a "Library" tab.
- **Improved Chat:** A "Chat" tab featuring a chatbot interface with history, document selection, and source attribution.
- **Backend:** 
  - ChromaDB using persistent local storage.
  - Metadata tracking for document lifecycle.

## 3. Architecture
### 3.1 Storage Layer
- `./data/pdfs/`: Storage for the original PDF files.
- `./data/vector_db/`: Persistent ChromaDB storage.
- `./data/metadata.json`: JSON file tracking document info (name, collection_id, upload_date).

### 3.2 Data Flow
1. **Upload:** PDF -> `./data/pdfs/` -> Text Chunks -> ChromaDB (`./data/vector_db/`) -> Update `metadata.json`.
2. **Retrieval:** Select Document -> Load Chroma Collection -> User Query -> Similarity Search -> Context.
3. **Chat:** Context + Query -> LLM (Ollama/OpenAI) -> Response + Sources -> UI.

## 4. UI Design (Gradio)
- **Tab 1: My Library**
  - File upload component (triggering indexing).
  - Dataframe/Gallery showing existing documents with a "Delete" action.
- **Tab 2: AI Chat**
  - Dropdown to select the active document from the library.
  - `gr.Chatbot` for conversational history.
  - `gr.Markdown` area for source chunks.

## 5. Implementation Details
- **RAG Updates:** Modify `build_rag_chain` to accept a collection name and handle persistent loading.
- **State Management:** Use Gradio's state or global variables to track the current active document and its chain.
- **Cleanup:** Ensure deleting a document removes the file, the Chroma collection, and the metadata entry.

## 6. Success Criteria
- [ ] Application starts and loads existing documents without re-indexing.
- [ ] New PDFs can be uploaded and appear in the library list.
- [ ] User can switch between documents in the chat tab.
- [ ] Deleting a document removes it from the UI and disk.
- [ ] Chat history is maintained within a single session.
