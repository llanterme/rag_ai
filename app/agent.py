from typing import List, Dict, Any, Optional
import os
from pydantic_ai import Agent, RunContext
from langchain_openai import OpenAIEmbeddings

from utils.vector_store import PineconeStore
from config import TOP_K

# Initialize the RAG agent
rag_agent = Agent(
    'openai:gpt-4o',
    system_prompt=(
        "You are a helpful assistant that answers questions based on the provided context. "
        "Use the `query_knowledge_base` tool to search for relevant information. "
        "Always cite your sources and be honest when you don't know something. "
        "If the retrieved context doesn't contain relevant information, say so clearly."
    ),
)

# Initialize vector store
vector_store = PineconeStore()

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings()

@rag_agent.tool
async def query_knowledge_base(ctx: RunContext, query: str) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for information relevant to the query.
    
    Args:
        query: The search query
        
    Returns:
        A list of relevant document chunks with their metadata
    """
    # Generate embedding for the query
    query_embedding = embeddings_model.embed_query(query)
    
    # Query Pinecone
    results = vector_store.query(query_embedding, top_k=TOP_K)
    
    # Format results
    formatted_results = []
    for i, match in enumerate(results):
        formatted_results.append({
            "content": match["metadata"]["text"],
            "metadata": {
                "source": match["metadata"].get("filename", "Unknown"),
                "score": match["score"]
            }
        })
    
    return formatted_results

@rag_agent.tool
async def generate_answer(ctx: RunContext, question: str, snippets: List[Dict[str, Any]]) -> str:
    """
    Generate an answer to the question based on the retrieved snippets.
    
    Args:
        question: The user's question
        snippets: Retrieved document snippets
        
    Returns:
        A comprehensive answer with source attribution
    """
    # Format snippets for the prompt
    formatted_snippets = ""
    for i, snippet in enumerate(snippets):
        formatted_snippets += f"Snippet {i+1} (Source: {snippet['metadata']['source']}):\n{snippet['content']}\n\n"
    
    # Create prompt
    prompt = f"""
    You are a helpful assistant. Use the following retrieved snippets to answer the question:
    
    {formatted_snippets}
    
    Question: {question}
    
    Answer:
    """
    
    # Use the LLM to generate an answer
    # Note: We're using the agent's own model to generate the answer
    # This is just a helper function to format the prompt
    return prompt

class RAGAgentWrapper:
    """Wrapper for the RAG agent to simplify usage in the Streamlit app."""
    
    def __init__(self):
        self.agent = rag_agent
    
    async def query(self, question: str):
        """
        Query the RAG agent with a question.
        
        Args:
            question: The user's question
            
        Returns:
            The agent's response
        """
        return await self.agent.arun(question)
    
    def query_sync(self, question: str):
        """
        Synchronous version of query.
        
        Args:
            question: The user's question
            
        Returns:
            The agent's response
        """
        return self.agent.run_sync(question)
