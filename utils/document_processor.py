import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import PyPDF2
from tqdm import tqdm
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
    """Read content from a text file in chunks to be memory efficient."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            # Read and process in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            chunks = []
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
            return ''.join(chunks)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return ""

def read_pdf_file(file_path: Path) -> str:
    """Extract text from a PDF file with better error handling and progress feedback."""
    try:
        text_parts = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Process pages with progress bar
            for page_num in tqdm(range(total_pages), desc=f"Processing {file_path.name}"):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text + "\n\n")
                except Exception as e:
                    print(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
                    
        return ''.join(text_parts)
    except Exception as e:
        print(f"Error reading PDF {file_path}: {str(e)}")
        return ""

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
    """Generate embeddings for a list of chunks with better batching and error handling."""
    embeddings = OpenAIEmbeddings()
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        return []
    
    # Process in parallel batches
    BATCH_SIZE = 100  # Increased batch size for better throughput
    NUM_WORKERS = 4   # Number of parallel workers
    
    def process_batch(batch_indices):
        batch_chunks = [chunks[i] for i in batch_indices]
        batch_texts = [chunk.text for chunk in batch_chunks]
        try:
            batch_embeddings = embeddings.embed_documents(batch_texts)
            for i, embedding in zip(batch_indices, batch_embeddings):
                chunks[i].embedding = embedding
            return len(batch_indices)
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return 0
    
    # Create batches of indices
    batch_indices = [range(i, min(i + BATCH_SIZE, total_chunks)) 
                    for i in range(0, total_chunks, BATCH_SIZE)]
    
    # Process batches in parallel
    completed = 0
    with tqdm(total=total_chunks, desc="Generating embeddings") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batch_indices]
            for future in as_completed(futures):
                completed += future.result() or 0
                pbar.update(BATCH_SIZE)
    
    # Filter out any chunks that failed to get embeddings
    return [chunk for chunk in chunks if chunk.embedding is not None]

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
