import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

from app.agent import RAGAgentWrapper
from ingest import ingest_documents

async def test_agent():
    """Test the RAG agent with a sample question."""
    load_dotenv()
    
    # Create a test directory if it doesn't exist
    test_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data"))
    test_dir.mkdir(exist_ok=True)
    
    # Create a sample text file for testing
    sample_text = """
    # RAG Systems
    
    Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches 
    for natural language processing tasks. It enhances large language models by retrieving relevant information 
    from external knowledge sources before generating a response.
    
    ## Benefits of RAG
    
    1. **Improved Accuracy**: By grounding responses in retrieved information, RAG systems reduce hallucinations.
    2. **Up-to-date Knowledge**: RAG can access the latest information that wasn't available during model training.
    3. **Transparency**: Sources can be cited, making the system more trustworthy.
    
    ## Components of a RAG System
    
    - **Document Store**: Repository of documents to be searched.
    - **Retriever**: Component that finds relevant documents based on a query.
    - **Generator**: Language model that creates a response using the retrieved information.
    """
    
    test_file_path = test_dir / "rag_info.txt"
    with open(test_file_path, "w") as f:
        f.write(sample_text)
    
    # Ingest the test document
    print("Ingesting test document...")
    ingest_documents(file_paths=[str(test_file_path)])
    
    # Initialize the agent
    print("Initializing agent...")
    agent = RAGAgentWrapper()
    
    # Test query
    test_query = "What is RAG and what are its benefits?"
    print(f"\nQuery: {test_query}")
    
    # Get response
    print("Getting response...")
    response = await agent.query(test_query)
    
    # Print result
    print("\nAgent Response:")
    print("=" * 50)
    
    # Extract the text response from the agent result
    for message in response.new_messages():
        for part in message.parts:
            if part.part_kind == 'text':
                print(part.content)
    
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_agent())
