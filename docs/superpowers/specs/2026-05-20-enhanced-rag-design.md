# Multi-Document RAG with Memory and Citations - Design Doc

**Date:** 2026-05-20
**Status:** Approved

## Overview
This design upgrades the RAG Chatbot to support a ChatGPT-like experience: querying across a unified library of documents, maintaining conversation history, and providing verifiable source citations.

## Architectural Changes

### 1. Unified Document Retrieval
*   **Collection Strategy**: Migrate from per-PDF collections to a single, unified `library` ChromaDB collection.
*   **Metadata**: Documents will be stored with metadata: `{'source': filename, 'page': page_number}`.
*   **Benefit**: Enables semantic search across the entire knowledge base.

### 2. Conversational Memory
*   **Component**: Implement LangChain's `ChatMessageHistory` or `RunnableWithMessageHistory`.
*   **Workflow**: Chat interface will store `(user_message, ai_response)` pairs in session state.
*   **Context Passing**: The RAG chain will incorporate the last N messages to provide context to the LLM.

### 3. Source Citations
*   **Prompt Engineering**: Update the RAG system prompt to explicitly require the LLM to cite sources in the format `[source: filename.pdf | page: X]`.
*   **Rendering**: Backend will parse citations and format them as clickable links/references within the Gradio chatbot UI.

## Success Criteria
- User can upload multiple documents, all searchable in one chat session.
- Chatbot recalls previous questions/answers within the same session.
- Every AI response includes clear, linkable references to the source document and page.
- Responses stream token-by-token.
