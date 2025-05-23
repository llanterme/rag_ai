import os
import sys
import subprocess

def main():
    """Main entry point for the RAG AI application."""
    print("Starting RAG AI application...")
    
    # Add the current directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    
    # Run the Streamlit app
    subprocess.run(["streamlit", "run", os.path.join(current_dir, "app", "streamlit_app.py")])

if __name__ == "__main__":
    main()
