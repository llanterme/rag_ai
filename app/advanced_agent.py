from typing import List, Dict, Any, Optional
import os
from pydantic_ai import Agent, RunContext
from langchain_openai import OpenAIEmbeddings
import json

from utils.vector_store import PineconeStore
from config import TOP_K

# Initialize the advanced RAG agent with more capabilities
advanced_rag_agent = Agent(
    'openai:gpt-4o',
    system_prompt=(
        "You are a helpful research assistant that answers questions based on the provided context. "
        "Use the `query_knowledge_base` tool to search for relevant information. "
        "Always cite your sources with clear attribution. "
        "If the retrieved context doesn't contain relevant information, be honest about it. "
        "You can use the `refine_search` tool to perform a more targeted search if needed. "
        "Format your responses in a clear, well-structured manner with markdown formatting when appropriate."
    ),
)

# Initialize vector store
vector_store = PineconeStore()

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings()

@advanced_rag_agent.tool
async def query_knowledge_base(ctx: RunContext, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for information relevant to the query.
    
    Args:
        query: The search query
        top_k: Number of results to return (default: 4)
        
    Returns:
        A list of relevant document chunks with their metadata
    """
    # Generate embedding for the query
    query_embedding = embeddings_model.embed_query(query)
    
    # Query Pinecone
    results = vector_store.query(query_embedding, top_k=top_k)
    
    # Format results
    formatted_results = []
    for i, match in enumerate(results):
        formatted_results.append({
            "content": match["metadata"]["text"],
            "metadata": {
                "source": match["metadata"].get("filename", "Unknown"),
                "score": match["score"],
                "section_title": match["metadata"].get("section_title", ""),
                "source_id": match["metadata"].get("source_id", "")
            }
        })
    
    return formatted_results

@advanced_rag_agent.tool
async def refine_search(ctx: RunContext, original_query: str, refined_query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Perform a refined search based on the original query and results.
    
    Args:
        original_query: The original search query
        refined_query: A more specific query based on initial results
        top_k: Number of results to return (default: 4)
        
    Returns:
        A list of relevant document chunks with their metadata
    """
    # Generate embedding for the refined query
    query_embedding = embeddings_model.embed_query(refined_query)
    
    # Query Pinecone
    results = vector_store.query(query_embedding, top_k=top_k)
    
    # Format results
    formatted_results = []
    for i, match in enumerate(results):
        formatted_results.append({
            "content": match["metadata"]["text"],
            "metadata": {
                "source": match["metadata"].get("filename", "Unknown"),
                "score": match["score"],
                "section_title": match["metadata"].get("section_title", ""),
                "source_id": match["metadata"].get("source_id", ""),
                "refined_query": refined_query
            }
        })
    
    return formatted_results

@advanced_rag_agent.tool
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
        source = snippet['metadata']['source']
        score = snippet['metadata'].get('score', 'N/A')
        section = snippet['metadata'].get('section_title', '')
        section_info = f" - Section: {section}" if section else ""
        
        formatted_snippets += f"Snippet {i+1} (Source: {source}{section_info}, Relevance: {score:.4f}):\n{snippet['content']}\n\n"
    
    # Create prompt
    prompt = f"""
    You are a helpful research assistant. Use the following retrieved snippets to answer the question:
    
    {formatted_snippets}
    
    Question: {question}
    
    Provide a comprehensive answer with clear source attribution. If the snippets don't contain enough information to answer the question fully, be honest about the limitations. Format your response using markdown for better readability.
    
    Answer:
    """
    
    # Use the LLM to generate an answer
    return prompt

class AdvancedRAGAgentWrapper:
    """Wrapper for the advanced RAG agent to simplify usage in the Streamlit app."""
    
    def __init__(self):
        self.agent = advanced_rag_agent
    
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
