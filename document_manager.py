#!/usr/bin/env python3
"""
Document Manager for RAG AI

This script provides a command-line interface to manage documents in the Pinecone vector store.
"""
import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.vector_store import PineconeStore

def list_documents(user_code: str):
    """
    List all ingested documents for a specific user.
    
    Args:
        user_code: The user's access code
    """
    if not user_code:
        print("Error: User code is required")
        return
        
    try:
        store = PineconeStore(user_code=user_code)
        documents = store.list_ingested_documents()
        
        if not documents:
            print("No documents found in the vector store.")
            return
            
        print("\n=== Ingested Documents ===")
        for i, doc in enumerate(documents, 1):
            print(f"{i}. {doc.get('filename', 'Unknown')} ({doc.get('chunk_count', 0)} chunks)")
        print("")
        return documents
    except Exception as e:
        error_msg = f"Error listing documents for user {user_code}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def add_documents(file_paths: List[Path], user_code: str) -> int:
    """
    Add documents to the vector store for a specific user.
    
    Args:
        file_paths: List of file paths to add
        user_code: The user's access code (required)
        
    Returns:
        Number of documents successfully added
    """
    if not user_code:
        raise ValueError("User code is required to add documents")
    if not file_paths:
        print("No files to process")
        return 0
        
    try:
        store = PineconeStore(user_code=user_code)
        # Ensure the index exists before trying to add documents
        if not store.ensure_index_exists():
            store.create_index()
            
        count = store.ingest_documents(file_paths)
        print(f"Added {count} documents for user {user_code}")
        return count
    except Exception as e:
        error_msg = f"Error adding documents for user {user_code}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def delete_document(file_path: Path, user_code: str) -> bool:
    """
    Delete a document from the vector store for a specific user.
    
    Args:
        file_path: Path to the document to delete
        user_code: The user's access code (required)
        
    Returns:
        True if deletion was successful, False otherwise
    """
    if not user_code:
        raise ValueError("User code is required to delete documents")
        
    try:
        store = PineconeStore(user_code=user_code)
        success = store.delete_document(file_path)
        if success:
            print(f"Deleted document {file_path} for user {user_code}")
        else:
            print(f"Document {file_path} not found for user {user_code}")
        return success
    except Exception as e:
        error_msg = f"Error deleting document {file_path} for user {user_code}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def update_document(file_path: Path, user_code: str) -> bool:
    """
    Update a document in the vector store for a specific user.
    
    Args:
        file_path: Path to the document to update
        user_code: The user's access code (required)
        
    Returns:
        True if update was successful, False otherwise
    """
    if not user_code:
        raise ValueError("User code is required to update documents")
        
    try:
        store = PineconeStore(user_code=user_code)
        success = store.update_document(file_path)
        if success:
            print(f"Updated document {file_path} for user {user_code}")
        else:
            print(f"Failed to update document {file_path} for user {user_code}")
        return success
    except Exception as e:
        error_msg = f"Error updating document {file_path} for user {user_code}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def clear_all_documents(user_code: str) -> bool:
    """
    Remove all documents from the vector store for a specific user.
    
    Args:
        user_code: The user's access code (required)
        
    Returns:
        True if operation was successful, False otherwise
    """
    if not user_code:
        raise ValueError("User code is required to clear documents")
        
    try:
        # Ask for confirmation
        confirm = input(f"Are you sure you want to delete ALL documents for user {user_code}? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled")
            return False
            
        store = PineconeStore(user_code=user_code)
        success = store.clear_all_documents()
        if success:
            print(f"Cleared all documents for user {user_code}")
        else:
            print(f"Failed to clear documents for user {user_code}")
        return success
    except Exception as e:
        error_msg = f"Error clearing documents for user {user_code}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

def main():
    parser = argparse.ArgumentParser(description="Manage documents in the RAG AI vector store.")
    parser.add_argument('--user-code', required=True, help='User access code (4-20 alphanumeric characters)')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all ingested documents')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add documents to the vector store')
    add_group = add_parser.add_mutually_exclusive_group(required=True)
    add_group.add_argument('--dir', help='Directory containing documents to add')
    add_group.add_argument('--files', nargs='+', help='List of files to add')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a document from the vector store')
    delete_parser.add_argument('file', help='Path to the document to delete')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update a document in the vector store')
    update_parser.add_argument('file', help='Path to the document to update')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all documents from the vector store')
    
    args = parser.parse_args()
    
    # Validate user code format
    if not re.match(r'^[a-zA-Z0-9]{4,20}$', args.user_code):
        print("Error: User code must be 4-20 alphanumeric characters", file=sys.stderr)
        sys.exit(1)
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'list':
            list_documents(args.user_code)
        elif args.command == 'add':
            if args.dir:
                file_paths = list(Path(args.dir).glob('*'))
                if not file_paths:
                    print(f"No files found in directory: {args.dir}", file=sys.stderr)
                    return 1
            else:
                file_paths = [Path(f) for f in args.files]
            
            # Filter out directories and non-existent files
            valid_files = [f for f in file_paths if f.is_file()]
            if not valid_files:
                print("Error: No valid files found to process", file=sys.stderr)
                return 1
                
            add_documents(valid_files, args.user_code)
        elif args.command == 'delete':
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                return 1
            delete_document(file_path, args.user_code)
        elif args.command == 'update':
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                return 1
            update_document(file_path, args.user_code)
        elif args.command == 'clear':
            clear_all_documents(args.user_code)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
