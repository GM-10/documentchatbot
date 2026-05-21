# Unified RAG Library, Memory, and Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the RAG system to support multi-document retrieval (single collection), conversational memory, and source citations.

**Architecture:**
- **Retrieval**: Migrate from per-PDF collections to one `library` collection in ChromaDB.
- **Memory**: Integrate `ChatMessageHistory` into the chat loop.
- **Citations**: Prompt the LLM for citation formatting and parse/render in the UI.

**Tech Stack:** Python, LangChain, ChromaDB, Gradio.

---

### Task 1: Migrate to Unified Library Collection

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Update `index_pdf` to use single `library` collection**
Modify `index_pdf` to use `collection_name="library"` for all documents. Update `build_rag_chain` to retrieve from the same collection.

- [ ] **Step 2: Update retrieval logic**
Update retrieval to filter by source filename if specific document selection is still needed, or allow full library search.

### Task 2: Implement Conversational Memory

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Add memory to RAG chain**
Replace the simple retrieval chain with a conversational one that accepts history.

### Task 3: Implement Citations and Streaming

**Files:**
- Modify: `App.py`

- [ ] **Step 1: Update prompt for citations**
Inject a system prompt instruction: "Always cite sources as [source: filename.pdf | page: X]".

- [ ] **Step 2: Implement Streaming**
Update `chat_respond` to use `chain.stream()` and yield responses to the Gradio chatbot UI.

### Task 4: Verification

- [ ] **Step 1: Functional Tests**
Upload multiple PDFs, ask a question covering multiple docs, verify memory (follow-up question), verify citations, verify streaming.
