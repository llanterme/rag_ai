## Project Planning: RAG AI Agent with Pydantic AI and Pinecone

### Project Overview

Build a modular Retrieval-Augmented Generation (RAG) AI agent using Pydantic AI. The agent will query a Pinecone-based knowledge base populated by local text and PDF files. Users can upload documents, trigger ingestion, and chat with the agent in a Streamlit UI.

### Core Components

1. **Document Ingestion Pipeline**

   * **Function**: `ingest_documents(path: Path) -> List[Chunk]`
   * Load TXT/PDF files from a folder or upload.
   * Compute a SHA256 hash for each file to support incremental ingest.
   * Chunk text into 500–1 000 token segments with overlap.
   * Generate embeddings (via OpenAIEmbeddings).
   * Upsert new chunks into Pinecone only if file-hash is not already ingested.
   * Maintain a local `ingested_files.json` manifest (file-hashes, timestamps).

2. **Pinecone Index**

   * **Index Name**: `pdf-chat`
   * **Namespace**: versioned (e.g. `pdfs_v1`) to allow future breaking changes.
   * **Setup**: Created on first run if missing, with cosine similarity and Serverless spec.
   * **Data**: Stores chunk embeddings + metadata (`text`, `source_id`, optional `filename`, `page`).
   * **Search**: Efficient semantic retrieval via Pinecone’s ANN engine.

3. **Pydantic AI Agent**

   * **Tools**:

     * `query_knowledge_base(query: str) -> List[Document]`: wraps Pinecone `.query()`.
     * `generate_answer(question: str, snippets: List[Document]) -> str`: invokes OpenAI LLM with a prompt template.
   * **Prompt Template**:

     ```text
     You are a helpful assistant. Use the following retrieved snippets to answer the question:
     {snippets}
     Question: {question}
     Answer:
     ```
   * Decouple retriever and generator for independent testing and swapping.

4. **Streamlit User Interface**

   * **Upload Page**: File uploader widget → save to `~/pdf/` → call `ingest_documents()` → show progress and results.
   * **Chat Page**: Text input + Send button → call Pydantic AI agent → display response and toggleable source snippets.
   * Protect with basic auth or Streamlit’s password feature for non-local deployments.

### Technology Stack

* **Language**: Python 3.11+
* **AI Framework**: Pydantic AI for agent orchestration
* **Vector DB**: Pinecone
* **Embeddings**: OpenAI Embeddings API
* **LLM Provider**: OpenAI (GPT‑4.1 mini or similar)
* **UI**: Streamlit
* **PDF Processing**: PyPDF2 or `langchain-document-loaders`

### Environment & Configuration

* **`.env.example`**:

  ```ini
  OPENAI_API_KEY=
  PINECONE_API_KEY=
  PINECONE_ENV=us-west1-gcp
  PINECONE_INDEX_NAME=pdf-chat
  PINECONE_NAMESPACE=pdfs_v1
  CHUNK_SIZE=1000
  CHUNK_OVERLAP=200
  TOP_K=4
  ```
* Centralize constants (chunk size, overlap, top-k) in `config.py` or via env vars.


### Security & UX

* **File Validation**: Accept only `.pdf` and `.txt`; enforce size limit.
* **Error Handling**: Gracefully report ingest or query failures in the UI.


*Mark tasks complete in `task.md` as you progress.* 
