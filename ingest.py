import os
import argparse
from pathlib import Path
from typing import List

from utils.vector_store import PineconeStore
from utils.document_processor import load_ingested_hashes

def ingest_documents(directory_path: str = None, file_paths: List[str] = None) -> int:
    """
    Ingest documents into Pinecone.
    
    Args:
        directory_path: Path to directory containing documents to ingest
        file_paths: List of specific file paths to ingest
        
    Returns:
        Number of new documents ingested
    """
    store = PineconeStore()
    
    # Collect files to process
    files_to_process = []
    
    if directory_path:
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find all PDF and TXT files in the directory
        for ext in ['.pdf', '.txt']:
            files_to_process.extend(list(dir_path.glob(f"**/*{ext}")))
    
    if file_paths:
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"Warning: File does not exist: {file_path}")
                continue
            if path.suffix.lower() not in ['.pdf', '.txt']:
                print(f"Warning: Unsupported file type: {path.suffix}")
                continue
            files_to_process.append(path)
    
    if not files_to_process:
        print("No files to process.")
        return 0
    
    # Get already ingested files
    ingested_hashes = load_ingested_hashes()
    
    # Ingest documents
    num_ingested = store.ingest_documents(files_to_process)
    
    # Print summary
    print(f"Processed {len(files_to_process)} files.")
    print(f"Ingested {num_ingested} new documents.")
    print(f"Total documents in knowledge base: {len(ingested_hashes) + num_ingested}")
    
    return num_ingested

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="Directory containing documents to ingest")
    group.add_argument("--files", type=str, nargs="+", help="Specific files to ingest")
    
    args = parser.parse_args()
    
    if args.dir:
        ingest_documents(directory_path=args.dir)
    elif args.files:
        ingest_documents(file_paths=args.files)

if __name__ == "__main__":
    main()
