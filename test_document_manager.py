#!/usr/bin/env python3
"""
Test script for document management functionality.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent))

from document_manager import (
    list_documents,
    add_documents,
    delete_document,
    update_document,
    clear_all_documents
)

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

def test_document_management():
    """Test document management functionality."""
    # Create a test file
    test_file = Path("test_document.txt")
    try:
        # Create a test file
        with open(test_file, "w") as f:
            f.write("This is a test document. " * 50)  # Make it long enough to be chunked
        
        # Test adding a document
        print_header("TESTING DOCUMENT ADDITION")
        add_documents([test_file])
        
        # List documents
        print_header("LISTING DOCUMENTS")
        list_documents()
        
        # Test updating the document
        print_header("TESTING DOCUMENT UPDATE")
        with open(test_file, "a") as f:
            f.write("\n\nThis is an update to the test document.")
        update_document(test_file)
        
        # List documents again
        print_header("LISTING DOCUMENTS AFTER UPDATE")
        list_documents()
        
        # Test deleting the document
        print_header("TESTING DOCUMENT DELETION")
        delete_document(test_file)
        
        # List documents again
        print_header("LISTING DOCUMENTS AFTER DELETION")
        list_documents()
        
        print("\nAll tests completed successfully!")
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    print("Testing Document Management")
    print("=" * 50)
    
    # Clear all documents first to start with a clean slate
    print("Clearing all documents...")
    clear_all_documents()
    
    # Run tests
    test_document_management()
