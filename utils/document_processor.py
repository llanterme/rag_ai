import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import CHUNK_SIZE, CHUNK_OVERLAP, INGESTED_FILES_PATH

class Chunk:
    """Represents a text chunk with its metadata and embedding."""
    def __init__(self, 
                 text: str, 
                 metadata: Dict[str, Any], 
                 embedding: Optional[List[float]] = None):
        self.text = text
        self.metadata = metadata
        self.embedding = embedding

def file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()

def load_ingested_hashes() -> Set[str]:
    """Read the set of already-ingested file hashes from disk."""
    if os.path.exists(INGESTED_FILES_PATH):
        with open(INGESTED_FILES_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_ingested_hashes(hashes: Set[str]):
    """Persist the updated set of ingested file hashes to disk."""
    with open(INGESTED_FILES_PATH, "w") as f:
        json.dump(list(hashes), f)

def read_text_file(file_path: Path) -> str:
    """Read content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path: Path) -> str:
    """Extract text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    return text

def chunk_text(text: str, source_id: str, filename: str) -> List[Chunk]:
    """Split text into chunks with metadata."""
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )
    
    texts = splitter.split_text(text)
    chunks = []
    
    for i, chunk_text in enumerate(texts):
        metadata = {
            "source_id": source_id,
            "filename": filename,
            "chunk_id": i
        }
        chunks.append(Chunk(text=chunk_text, metadata=metadata))
    
    return chunks

def generate_embeddings(chunks: List[Chunk]) -> List[Chunk]:
    """Generate embeddings for a list of chunks."""
    embeddings = OpenAIEmbeddings()
    texts = [chunk.text for chunk in chunks]
    
    # Process in batches to avoid token limits
    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_texts = [chunk.text for chunk in batch]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        
        for j, embedding in enumerate(batch_embeddings):
            chunks[i+j].embedding = embedding
    
    return chunks

def process_file(file_path: Path) -> List[Chunk]:
    """Process a single file and return chunks with embeddings."""
    file_id = file_hash(file_path)
    filename = file_path.name
    
    # Extract text based on file type
    if file_path.suffix.lower() == '.pdf':
        text = read_pdf_file(file_path)
    elif file_path.suffix.lower() == '.txt':
        text = read_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    # Create chunks
    chunks = chunk_text(text, file_id, filename)
    
    # Generate embeddings
    chunks = generate_embeddings(chunks)
    
    return chunks
