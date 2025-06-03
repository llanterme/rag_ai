from typing import List, Dict, Any, Optional, Set, Tuple
from pinecone import Pinecone, ServerlessSpec
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import (
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    DATA_DIR
)
from utils.document_processor import Chunk, load_ingested_hashes, save_ingested_hashes, process_file, file_hash
from utils.user_manager import UserManager

class PineconeStore:
    """Handles interactions with Pinecone vector database with support for user-specific indexes."""
    
    def __init__(self, user_code: str):
        """
        Initialize PineconeStore for a specific user.
        
        Args:
            user_code: User code to use for user-specific operations.
        """
        if not user_code:
            raise ValueError("User code is required")
            
        self.client = Pinecone(api_key=PINECONE_API_KEY)
        self.user_manager = UserManager()
        
        # Get user-specific configuration
        user_config = self.user_manager.get_user_index(user_code)
        if not user_config:
            # If user doesn't exist, create a new user config with a default index name
            # Format: rag_ai_user_<first_8_chars_of_hash>
            import hashlib
            hash_suffix = hashlib.sha256(user_code.encode()).hexdigest()[:8]
            default_index_name = f"rag_ai_user_{hash_suffix}"
            
            try:
                user_config = self.user_manager.create_user_index(
                    index_name=default_index_name,
                    access_code=user_code
                )
                if not user_config:
                    raise ValueError(f"Failed to create user with code: {user_code}")
            except Exception as e:
                raise ValueError(f"Failed to create user index: {str(e)}")
            
        self.index_name = user_config["index_name"]
        self.namespace = f"user_{user_code}"
        self.document_chunks_file = self.user_manager.get_document_chunks_file(user_code)
        
        print(f"PineconeStore initialized with index_name='{self.index_name}', namespace='{self.namespace}'")
        self.index = None
        self._load_document_chunks()
        
    def create_index(self):
        """Create the Pinecone index if it doesn't exist."""
        import logging
        import traceback
        logger = logging.getLogger("vector_store")
        
        logger.info(f"In create_index for index '{self.index_name}'")
        
        try:
            existing_indexes = self.client.list_indexes().names()
            logger.info(f"Existing indexes: {existing_indexes}")
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.client.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENV
                    )
                )
                logger.info(f"Successfully created new index '{self.index_name}'")
            else:
                logger.info(f"Index '{self.index_name}' already exists, skipping creation")
                
            logger.info(f"Connecting to index '{self.index_name}'")
            self.index = self.client.Index(self.index_name)
            logger.info(f"Successfully connected to index '{self.index_name}'")
            return self.index
        except Exception as e:
            error_msg = f"Error in create_index for '{self.index_name}': {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return None
        
    def _load_document_chunks(self) -> Dict[str, Set[str]]:
        """Load the document to chunk mappings from disk."""
        if os.path.exists(self.document_chunks_file):
            try:
                with open(self.document_chunks_file, 'r') as f:
                    return {file_id: set(chunk_ids) 
                            for file_id, chunk_ids in json.load(f).items()}
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_document_chunks(self, document_chunks: Dict[str, Set[str]]):
        """Save the document to chunk mappings to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.document_chunks_file), exist_ok=True)
        
        with open(self.document_chunks_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable = {file_id: list(chunk_ids) 
                          for file_id, chunk_ids in document_chunks.items()}
            json.dump(serializable, f, indent=2)
        
    def ensure_index_exists(self):
        """
        Check if the index exists and is accessible.
        If not, try to create it.
        
        Returns:
            bool: True if the index exists or was created successfully, False otherwise
        """
        import logging
        import traceback
        logger = logging.getLogger("vector_store")
        logger.setLevel(logging.INFO)
        
        # Add a handler if none exists
        if not logger.handlers:
            import sys
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Checking if index '{self.index_name}' exists")
        
        try:
            logger.info("Listing available Pinecone indexes")
            existing_indexes = self.client.list_indexes().names()
            logger.info(f"Available indexes: {existing_indexes}")
            
            if self.index_name in existing_indexes:
                logger.info(f"Index '{self.index_name}' found, initializing connection")
                self.index = self.client.Index(self.index_name)
                logger.info(f"Successfully connected to index '{self.index_name}'")
                return True
                
            # If we get here, the index doesn't exist - create it
            logger.warning(f"Index '{self.index_name}' not found. Creating new index...")
            success = self.create_index()
            if success:
                logger.info(f"Successfully created index '{self.index_name}'")
                return True
            else:
                logger.error(f"Failed to create index '{self.index_name}'")
                return False
            
        except Exception as e:
            error_msg = f"Error checking/creating index '{self.index_name}': {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False
    
    def upsert_chunks(self, chunks: List[Chunk]) -> Tuple[int, Set[str]]:
        """
        Upsert chunks into Pinecone with better batching and error handling.
        Returns a tuple of (number of chunks upserted, set of source file IDs).
        """
        if not chunks:
            return 0, set()
            
        vectors = []
        source_ids = set()
        
        # Track which chunks belong to which documents
        document_chunks = self._load_document_chunks()
        
        # Prepare vectors with progress tracking
        with tqdm(total=len(chunks), desc="Preparing vectors") as pbar:
            for i, chunk in enumerate(chunks):
                if not chunk.embedding:
                    print(f"Warning: Chunk {i} has no embedding, skipping")
                    continue
                
                source_id = chunk.metadata['source_id']
                chunk_id = chunk.metadata['chunk_id']
                vector_id = f"{source_id}_{chunk_id}"
                
                # Track this chunk for the source document
                if source_id not in document_chunks:
                    document_chunks[source_id] = set()
                document_chunks[source_id].add(vector_id)
                source_ids.add(source_id)
                
                metadata = chunk.metadata.copy()
                metadata["text"] = chunk.text
                vectors.append((vector_id, chunk.embedding, metadata))
                pbar.update(1)
        
        if not vectors:
            return 0, set()
        
        # Save the updated document chunks mapping
        self._save_document_chunks(document_chunks)
        
        # Batch upsert with parallel processing
        BATCH_SIZE = 200  # Increased batch size for better throughput
        NUM_WORKERS = 4   # Number of parallel workers
        
        def process_batch(batch):
            try:
                self.index.upsert(vectors=batch, namespace=self.namespace)
                return len(batch)
            except Exception as e:
                print(f"Error upserting batch: {str(e)}")
                return 0
        
        # Process in parallel batches
        completed = 0
        with tqdm(total=len(vectors), desc="Uploading to Pinecone") as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Create batches
                batches = [vectors[i:i + BATCH_SIZE] 
                         for i in range(0, len(vectors), BATCH_SIZE)]
                
                # Submit all batches
                futures = [executor.submit(process_batch, batch) 
                          for batch in batches]
                
                # Process results as they complete
                for future in as_completed(futures):
                    completed += future.result() or 0
                    pbar.update(BATCH_SIZE)
        
        return completed, source_ids
    
    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: The embedding vector to query with
            top_k: Number of results to return
            
        Returns:
            List of document matches with metadata
        """
        import logging
        import traceback
        logger = logging.getLogger("vector_store")
        logger.setLevel(logging.INFO)
        
        # Add a handler if none exists
        if not logger.handlers:
            import sys
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Starting query on index '{self.index_name}' in namespace '{self.namespace}' with top_k={top_k}")
        
        # Ensure index exists and is accessible
        if not self.ensure_index_exists():
            logger.error(f"Index {self.index_name} does not exist. No documents to query.")
            return []
            
        try:
            # Get the index if not already loaded
            if self.index is None:
                logger.info(f"Loading index '{self.index_name}'")
                self.index = self.client.Index(self.index_name)
                
            logger.info(f"Executing query against index '{self.index_name}' in namespace '{self.namespace}'")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            
            # Log query results
            match_count = len(results.matches) if hasattr(results, 'matches') else 0
            logger.info(f"Query returned {match_count} matches")
            
            # Format results
            formatted_results = []
            if hasattr(results, 'matches'):
                for i, match in enumerate(results.matches):
                    try:
                        match_id = match.id if hasattr(match, 'id') else 'unknown'
                        match_score = match.score if hasattr(match, 'score') else 0.0
                        match_metadata = match.metadata if hasattr(match, 'metadata') else {}
                        
                        # Log each match
                        logger.info(f"Match {i}: id={match_id}, score={match_score}, metadata_keys={list(match_metadata.keys()) if isinstance(match_metadata, dict) else 'None'}")
                        
                        text = ''
                        if isinstance(match_metadata, dict) and 'text' in match_metadata:
                            text = match_metadata['text']
                            logger.info(f"Match {i} text length: {len(text)}")
                        
                        formatted_results.append({
                            'id': match_id,
                            'score': match_score,
                            'metadata': match_metadata,
                            'text': text
                        })
                    except Exception as e:
                        logger.error(f"Error processing match {i}: {str(e)}")
                        logger.error(traceback.format_exc())
            else:
                logger.warning(f"Query results has no 'matches' attribute. Results type: {type(results)}")
                
            logger.info(f"Formatted {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            error_msg = f"Error querying index {self.index_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return []
            
        return formatted_results
    
    def delete_document(self, file_path: Path) -> bool:
        """
        Delete all chunks associated with a document.
        Returns True if successful, False otherwise.
        """
        try:
            file_id = file_hash(file_path)
            document_chunks = self._load_document_chunks()
            
            if file_id not in document_chunks:
                print(f"No chunks found for file: {file_path}")
                return False
                
            # Delete all chunks for this document
            chunk_ids = list(document_chunks[file_id])
            
            # Delete in batches
            BATCH_SIZE = 1000
            for i in range(0, len(chunk_ids), BATCH_SIZE):
                batch = chunk_ids[i:i+BATCH_SIZE]
                self.index.delete(ids=batch, namespace=self.namespace)
            
            # Update document chunks
            del document_chunks[file_id]
            self._save_document_chunks(document_chunks)
            
            # Update ingested files
            ingested_hashes = load_ingested_hashes()
            if file_id in ingested_hashes:
                ingested_hashes.remove(file_id)
                save_ingested_hashes(ingested_hashes)
            
            return True
            
        except Exception as e:
            print(f"Error deleting document {file_path}: {str(e)}")
            return False
    
    def update_document(self, file_path: Path) -> bool:
        """
        Update a document by first deleting its chunks and then re-ingesting it.
        Returns True if successful, False otherwise.
        """
        try:
            # First delete existing chunks
            if not self.delete_document(file_path):
                print(f"Failed to delete existing chunks for {file_path}")
                return False
            
            # Then re-ingest the document
            chunks = process_file(file_path)
            if not chunks:
                print(f"No chunks generated for {file_path}")
                return False
                
            num_upserted, _ = self.upsert_chunks(chunks)
            if num_upserted == 0:
                print(f"Failed to upsert any chunks for {file_path}")
                return False
                
            # Update ingested files
            file_id = file_hash(file_path)
            ingested_hashes = load_ingested_hashes()
            ingested_hashes.add(file_id)
            save_ingested_hashes(ingested_hashes)
            
            return True
            
        except Exception as e:
            print(f"Error updating document {file_path}: {str(e)}")
            return False
    
    def list_ingested_documents(self) -> List[Dict[str, Any]]:
        """
        List all ingested documents with their metadata.
        Returns a list of dictionaries containing document information.
        """
        document_chunks = self._load_document_chunks()
        documents = []
        
        for file_id, chunk_ids in document_chunks.items():
            if not chunk_ids:
                continue
                
            # Get the first chunk to extract metadata
            try:
                result = self.index.fetch(ids=[next(iter(chunk_ids))], namespace=self.namespace)
                if result.vectors:
                    first_chunk = next(iter(result.vectors.values()))
                    metadata = first_chunk.metadata
                    documents.append({
                        'file_id': file_id,
                        'filename': metadata.get('filename', 'unknown'),
                        'chunk_count': len(chunk_ids),
                        'first_chunk_text': metadata.get('text', '')[:200] + '...'  # First 200 chars
                    })
            except Exception as e:
                print(f"Error fetching document info for {file_id}: {str(e)}")
        
        return documents
    
    def clear_all_documents(self) -> bool:
        """
        Delete all documents from the vector store.
        Returns True if successful, False otherwise.
        """
        try:
            # Delete all vectors in the namespace
            self.index.delete(delete_all=True, namespace=self.namespace)
            
            # Clear document chunks
            self._save_document_chunks({})
            
            # Clear ingested files
            save_ingested_hashes(set())
            
            return True
            
        except Exception as e:
            print(f"Error clearing all documents: {str(e)}")
            return False
            
    def ingest_documents(self, file_paths: List[Path]) -> int:
        """
        Ingest multiple documents into the vector store.
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            int: Number of documents successfully ingested
        """
        if not file_paths:
            return 0
            
        ingested_hashes = load_ingested_hashes()
        new_document_count = 0
        
        for file_path in file_paths:
            try:
                file_path = Path(file_path)
                file_id = file_hash(file_path)
                
                # Skip if already ingested
                if file_id in ingested_hashes:
                    print(f"Document already ingested: {file_path}")
                    continue
                    
                # Process the file
                chunks = process_file(file_path)
                if not chunks:
                    print(f"No chunks generated for {file_path}")
                    continue
                    
                # Upsert chunks
                num_upserted, _ = self.upsert_chunks(chunks)
                if num_upserted > 0:
                    ingested_hashes.add(file_id)
                    new_document_count += 1
                    print(f"Successfully ingested {file_path} with {num_upserted} chunks")
                else:
                    print(f"Failed to ingest {file_path}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Save the updated list of ingested files
        if new_document_count > 0:
            save_ingested_hashes(ingested_hashes)
            
        return new_document_count
