#!/usr/bin/env python3
"""
Document Manager for RAG AI

This script provides a command-line interface to manage documents in the Pinecone vector store.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from utils.vector_store import PineconeStore

def list_documents():
    """List all ingested documents."""
    store = PineconeStore()
    documents = store.list_ingested_documents()
    
    if not documents:
        print("No documents found in the vector store.")
        return
        
    print("\n=== Ingested Documents ===")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. Document ID: {doc['file_id']}")
        print(f"   Chunks: {doc['chunk_count']}")
        if doc['chunk_ids']:
            print(f"   Sample chunk IDs: {', '.join(doc['chunk_ids'][:3])}...")
    print("")

def add_documents(file_paths: List[Path]):
    """Add documents to the vector store."""
    store = PineconeStore()
    
    # Filter out non-existent files
    valid_paths = []
    for path in file_paths:
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
        valid_paths.append(path)
    
    if not valid_paths:
        print("No valid files to process.")
        return
    
    print(f"Processing {len(valid_paths)} documents...")
    count = store.ingest_documents(valid_paths)
    print(f"Successfully ingested {count} new documents.")

def delete_document(file_path: Path):
    """Delete a document from the vector store."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False
        
    store = PineconeStore()
    if store.delete_document(file_path):
        print(f"Successfully deleted document: {file_path}")
        return True
    else:
        print(f"Document not found in vector store: {file_path}")
        return False

def update_document(file_path: Path):
    """Update a document in the vector store."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return False
        
    store = PineconeStore()
    if store.update_document(file_path):
        print(f"Successfully updated document: {file_path}")
        return True
    else:
        print(f"Failed to update document: {file_path}")
        return False

def clear_all_documents():
    """Remove all documents from the vector store."""
    store = PineconeStore()
    count = store.clear_all_documents()
    print(f"Successfully deleted all {count} documents from the vector store.")

def main():
    parser = argparse.ArgumentParser(description="Manage documents in the RAG AI vector store.")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all ingested documents')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add documents to the vector store')
    add_parser.add_argument('files', nargs='+', help='Files or directories to add')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a document from the vector store')
    delete_parser.add_argument('file', help='File to delete')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update a document in the vector store')
    update_parser.add_argument('file', help='File to update')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Remove all documents from the vector store')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_documents()
            
        elif args.command == 'add':
            file_paths = []
            for path_str in args.files:
                path = Path(path_str)
                if path.is_dir():
                    # Add all PDF and TXT files in the directory
                    file_paths.extend(path.glob('*.pdf'))
                    file_paths.extend(path.glob('*.txt'))
                else:
                    file_paths.append(path)
            
            if not file_paths:
                print("No valid PDF or TXT files found.")
                return
                
            add_documents(file_paths)
            
        elif args.command == 'delete':
            delete_document(Path(args.file))
            
        elif args.command == 'update':
            update_document(Path(args.file))
            
        elif args.command == 'clear':
            confirm = input("WARNING: This will delete ALL documents from the vector store. Continue? (y/N): ")
            if confirm.lower() == 'y':
                clear_all_documents()
            else:
                print("Operation cancelled.")
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
