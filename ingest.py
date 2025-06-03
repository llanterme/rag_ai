#!/usr/bin/env python3
import os
import argparse
import sys
from pathlib import Path
from typing import List

from document_manager import add_documents

def ingest_documents(directory_path: str = None, file_paths: List[str] = None, user_code: str = None) -> int:
    """
    Ingest documents into Pinecone for a specific user.
    
    Args:
        directory_path: Path to directory containing documents to ingest
        file_paths: List of specific file paths to ingest
        user_code: The user's access code (required)
        
    Returns:
        Number of new documents ingested
    """
    if not user_code:
        print("Error: User code is required")
        return 0
        
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
    add_documents(files_to_process, user_code)
    return len(files_to_process)

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Directory containing documents to ingest")
    group.add_argument("--files", nargs="+", help="List of files to ingest")
    parser.add_argument("--user-code", required=True, help="User access code")
    
    args = parser.parse_args()
    
    # Validate user code format (alphanumeric, 4-20 chars)
    import re
    if not re.match(r'^[a-zA-Z0-9]{4,20}$', args.user_code):
        print("Error: User code must be 4-20 alphanumeric characters")
        sys.exit(1)
    
    try:
        if args.dir:
            print(f"Ingesting documents from directory: {args.dir}")
            count = ingest_documents(directory_path=args.dir, user_code=args.user_code)
        else:
            print(f"Ingesting {len(args.files)} files")
            count = ingest_documents(file_paths=args.files, user_code=args.user_code)
            
        if count > 0:
            print(f"✅ Successfully ingested {count} new documents for user {args.user_code}.")
        else:
            print("ℹ️ No new documents were ingested. They may have been processed already.")
            
    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
