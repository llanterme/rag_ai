#!/usr/bin/env python3
import os
import argparse
import sys
from pathlib import Path
from typing import List

from document_manager import add_documents

def ingest_documents(directory_path: str = None, file_paths: List[str] = None) -> int:
    """
    Ingest documents into Pinecone.
    
    Args:
        directory_path: Path to directory containing documents to ingest
        file_paths: List of specific file paths to ingest
        
    Returns:
        Number of new documents ingested
    """
    # Collect files to process
    files_to_process = []
    
    if directory_path:
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Error: Directory not found: {directory_path}")
            return 0
        
        # Add all PDF and text files from directory
        files_to_process.extend(dir_path.glob("*.pdf"))
        files_to_process.extend(dir_path.glob("*.txt"))
    
    if file_paths:
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                print(f"Warning: File not found: {path}")
                continue
            files_to_process.append(path)
    
    if not files_to_process:
        print("No files to process.")
        return 0
    
    # Use the document manager to add documents
    add_documents(files_to_process)
    return len(files_to_process)

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="Directory containing documents to ingest")
    group.add_argument("--files", type=str, nargs="+", help="Specific files to ingest")
    
    args = parser.parse_args()
    
    try:
        if args.dir:
            ingest_documents(directory_path=args.dir)
        elif args.files:
            ingest_documents(file_paths=args.files)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
