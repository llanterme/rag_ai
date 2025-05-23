from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import os
from pathlib import Path

from config import (
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE
)
from utils.document_processor import Chunk, load_ingested_hashes, save_ingested_hashes, process_file, file_hash

class PineconeStore:
    """Handles interactions with Pinecone vector database."""
    
    def __init__(self):
        self.client = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.namespace = PINECONE_NAMESPACE
        self.ensure_index_exists()
        self.index = self.client.Index(self.index_name)
        
    def ensure_index_exists(self):
        """Create the Pinecone index if it doesn't exist."""
        existing_indexes = self.client.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            self.client.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENV
                )
            )
    
    def upsert_chunks(self, chunks: List[Chunk]):
        """Upsert chunks into Pinecone."""
        vectors = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.embedding:
                raise ValueError(f"Chunk {i} has no embedding")
            
            metadata = chunk.metadata.copy()
            metadata["text"] = chunk.text
            
            vector_id = f"{chunk.metadata['source_id']}_{chunk.metadata['chunk_id']}"
            vectors.append((vector_id, chunk.embedding, metadata))
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
    
    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace
        )
        
        return results.matches
    
    def ingest_documents(self, file_paths: List[Path]) -> int:
        """
        Ingest documents into the vector store.
        Returns the number of new documents ingested.
        """
        # Load already ingested file hashes
        ingested_hashes = load_ingested_hashes()
        new_hashes = set()
        all_chunks = []
        
        # Process each file
        for file_path in file_paths:
            file_id = file_hash(file_path)
            
            # Skip if already ingested
            if file_id in ingested_hashes:
                continue
            
            # Process the file
            chunks = process_file(file_path)
            all_chunks.extend(chunks)
            new_hashes.add(file_id)
        
        # Upsert chunks to Pinecone
        if all_chunks:
            self.upsert_chunks(all_chunks)
        
        # Update ingested files record
        if new_hashes:
            ingested_hashes.update(new_hashes)
            save_ingested_hashes(ingested_hashes)
        
        return len(new_hashes)
