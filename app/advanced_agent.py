from typing import List, Dict, Any, Optional
import os
from pydantic_ai import Agent, RunContext
from langchain_openai import OpenAIEmbeddings

from utils.vector_store import PineconeStore
from config import TOP_K

class AdvancedRAGAgentWrapper:
    """Wrapper for the advanced RAG agent that requires a user code for initialization."""
    
    def __init__(self, user_code: str):
        """Initialize the advanced RAG agent with a user code.
        
        Args:
            user_code: The user's access code (required)
            
        Raises:
            ValueError: If user_code is not provided
        """
        if not user_code:
            raise ValueError("User code is required to initialize the advanced RAG agent")
        
        # Initialize the agent with system prompt
        self.agent = Agent(
            'openai:gpt-4o',
            system_prompt=(
                "You are a helpful research assistant that answers questions based on the provided context. "
                "Use the `query_knowledge_base` tool to search for relevant information. "
                "Always cite your sources with clear attribution. "
                "If the retrieved context doesn't contain relevant information, be honest about it. "
                "You can use the `refine_search` tool to perform a targeted search if needed. "
                "Format your responses in a clear, well-structured manner with markdown formatting when appropriate."
            ),
        )
        
        # Initialize vector store with user code
        self.vector_store = PineconeStore(user_code=user_code)
        self.embeddings_model = OpenAIEmbeddings()
        
        # Register tools
        self.agent.tool()(self._query_knowledge_base)
        self.agent.tool()(self._refine_search)
    
    async def _query_knowledge_base(self, ctx: RunContext, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for information relevant to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return (default: 4)
            
        Returns:
            A list of relevant document chunks with their metadata
        """
        # Generate embedding for the query
        query_embedding = self.embeddings_model.embed_query(query)
        
        # Query Pinecone
        results = self.vector_store.query(query_embedding, top_k=top_k)
        
        # Format results
        formatted_results = []
        for match in results:
            formatted_results.append({
                "content": match["metadata"]["text"],
                "metadata": {
                    "source": match["metadata"].get("filename", "Unknown"),
                    "score": match["score"]
                }
            })
        
        return formatted_results
    
    async def _refine_search(self, ctx: RunContext, original_query: str, refined_query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
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
        query_embedding = self.embeddings_model.embed_query(refined_query)
        
        # Query Pinecone with the refined query
        results = self.vector_store.query(query_embedding, top_k=top_k)
        
        # Format results
        formatted_results = []
        for match in results:
            formatted_results.append({
                "content": match["metadata"]["text"],
                "metadata": {
                    "source": match["metadata"].get("filename", "Unknown"),
                    "score": match["score"],
                    "original_query": original_query,
                    "refined_query": refined_query
                }
            })
        
        return formatted_results
    
    async def query(self, question: str) -> str:
        """
        Query the advanced RAG agent with a question.
        
        Args:
            question: The user's question
            
        Returns:
            The agent's response
        """
        # Use the agent to generate an answer
        response = await self.agent.run(
            f"""Answer the following question based on the available tools and context.
            If you need to search for information, use the query_knowledge_base tool.
            If you need to refine your search, use the refine_search tool.
            
            Question: {question}
            
            Please provide a comprehensive answer with source attribution."""
        )
        
        return response.text
    
    def query_sync(self, question: str) -> str:
        """
        Synchronous version of query.
        
        Args:
            question: The user's question
            
        Returns:
            The agent's response
        """
        import asyncio
        return asyncio.run(self.query(question))
