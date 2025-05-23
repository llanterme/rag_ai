#!/usr/bin/env python3
import os
import sys
import subprocess
from dotenv import load_dotenv

def check_environment():
    """Check if all required environment variables are set."""
    load_dotenv()
    
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENV",
        "PINECONE_INDEX_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file.")
        return False
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import streamlit
        import pydantic_ai
        import pinecone
        import openai
        import PyPDF2
        return True
    except ImportError as e:
        print(f"❌ Error: Missing dependency: {str(e)}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main entry point for running the RAG AI application."""
    print("🚀 Starting RAG AI application...")
    
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Add the current directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    # Check if data directory exists
    data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 Created data directory: {data_dir}")
    
    # Check if uploads directory exists
    uploads_dir = os.path.join(data_dir, "uploads")
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"📁 Created uploads directory: {uploads_dir}")
    
    print("✅ Environment check passed")
    print("✅ Dependencies check passed")
    print("✅ Directory structure check passed")
    
    print("\n🔍 Running Pinecone index check...")
    subprocess.run([sys.executable, os.path.join(current_dir, "check_pinecone.py")])
    
    print("\n🌐 Starting Streamlit app...")
    try:
        subprocess.run(["streamlit", "run", os.path.join(current_dir, "app", "streamlit_app.py")])
    except KeyboardInterrupt:
        print("\n👋 Shutting down RAG AI application...")
    except Exception as e:
        print(f"\n❌ Error running Streamlit app: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
