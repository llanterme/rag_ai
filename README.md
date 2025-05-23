# RAG AI Agent with Pydantic AI and Pinecone

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents, query information, and get AI-generated responses with source attribution.

## Features

- **Document Ingestion Pipeline**
  - Accepts local TXT and PDF files
  - Uses a simple chunking approach
  - Generates embeddings using OpenAI
  - Stores documents and vectors in Pinecone

- **Pydantic AI Agent**
  - Knowledge base search tool
  - OpenAI models for response generation
  - Context integration for accurate responses

- **Streamlit UI**
  - Document upload interface
  - Clean query interface
  - Response display with source attribution

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── agent.py          # Pydantic AI agent implementation
│   └── streamlit_app.py  # Streamlit UI
├── utils/
│   ├── __init__.py
│   ├── document_processor.py  # Document processing utilities
│   └── vector_store.py        # Pinecone vector store handler
├── data/                 # Directory for data storage
├── config.py             # Configuration settings
├── ingest.py             # Document ingestion script
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables**

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
PINECONE_INDEX_NAME=pdf-chat
PINECONE_NAMESPACE=pdfs
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
```

## Usage

### Running the application

```bash
python main.py
```

This will start the Streamlit application, which you can access in your browser.

### Ingesting documents from the command line

```bash
# Ingest documents from a directory
python ingest.py --dir path/to/documents

# Ingest specific files
python ingest.py --files path/to/file1.pdf path/to/file2.txt
```

## How It Works

1. **Document Ingestion**
   - Documents are processed and split into chunks
   - Each chunk is embedded using OpenAI embeddings
   - Embeddings and text are stored in Pinecone

2. **Query Processing**
   - User queries are embedded
   - Similar document chunks are retrieved from Pinecone
   - The Pydantic AI agent generates a response using the retrieved context

3. **Response Generation**
   - The agent combines retrieved information with its knowledge
   - Responses include source attribution
   - Users can view the source documents used to generate the response

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
