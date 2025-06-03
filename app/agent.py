from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from pydantic_ai import Agent
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

class RAGAgentWrapper:
    """Wrapper for the RAG agent that requires a user code for initialization."""
    
    def __init__(self, user_code: str):
        """Initialize the RAG agent with a user code.
        
        Args:
            user_code: The user's access code (required)
            
        Raises:
            ValueError: If user_code is not provided
        """
        if not user_code:
            raise ValueError("User code is required to initialize the RAG agent")
            
        self.agent = rag_agent
        self.vector_store = PineconeStore(user_code=user_code)
        self.embeddings_model = OpenAIEmbeddings()
    
    async def _query_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for information relevant to the query.
        
        Args:
            query: The search query
            
        Returns:
            A list of relevant document chunks with their metadata
        """
        import logging
        import traceback
        import json
        logger = logging.getLogger("agent")
        
        logger.info(f"Starting _query_knowledge_base with query: {query[:50]}...")
        
        try:
            # Generate embedding for the query
            logger.info("Generating embedding for query")
            query_embedding = self.embeddings_model.embed_query(query)
            logger.info(f"Generated embedding with length: {len(query_embedding)}")
            
            # Query Pinecone
            logger.info(f"Querying vector store with top_k={TOP_K}")
            results = self.vector_store.query(query_embedding, top_k=TOP_K)
            logger.info(f"Vector store query returned {len(results) if results else 0} results")
            
            if not results:
                logger.warning("No results returned from vector store")
                return []
                
            if not isinstance(results, list):
                logger.error(f"Invalid results format: {type(results)}")
                return []
                
            # Log the raw results for debugging
            try:
                logger.info(f"Raw results sample: {json.dumps(results[0]) if results else 'None'}")
            except:
                logger.info(f"Raw results first item: {results[0] if results else 'None'}")
                
            # Format results
            logger.info("Formatting results")
            formatted_results = []
            for i, match in enumerate(results):
                if not isinstance(match, dict):
                    logger.warning(f"Skipping invalid match format at index {i}: {type(match)}")
                    continue
                    
                try:
                    # Log the match structure
                    logger.info(f"Match {i} keys: {match.keys() if hasattr(match, 'keys') else 'No keys'}")
                    
                    metadata = match.get("metadata", {})
                    if metadata and not isinstance(metadata, dict):
                        logger.warning(f"Invalid metadata format: {type(metadata)}")
                        metadata = {}
                        
                    # Try to extract content from different possible locations
                    content = ""
                    if "text" in metadata:
                        content = metadata["text"]
                    elif "text" in match:
                        content = match["text"]
                    
                    if not content:
                        logger.warning(f"No content found in match {i}")
                        continue
                        
                    source = metadata.get("filename", "Unknown")
                    score = match.get("score", 0.0)
                    
                    logger.info(f"Match {i}: source={source}, score={score}, content_length={len(content)}")
                    
                    formatted_results.append({
                        "content": content,
                        "metadata": {
                            "source": source,
                            "score": score
                        }
                    })
                except Exception as e:
                    logger.error(f"Error formatting match {i}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            logger.info(f"Formatted {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def query(self, question: str) -> str:
        """
        Query the RAG agent with a question.
        
        Args:
            question: The user's question
            
        Returns:
            The agent's response
        """
        import logging
        import traceback
        logger = logging.getLogger("agent")
        logger.setLevel(logging.INFO)
        
        # Add a handler if none exists
        if not logger.handlers:
            import sys
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Starting RAGAgentWrapper.query with question: {question[:50]}...")
        
        try:
            # First, get relevant context
            logger.info("Querying knowledge base")
            snippets = await self._query_knowledge_base(question)
            logger.info(f"Retrieved {len(snippets)} snippets from knowledge base")
            
            if not snippets:
                logger.warning("No snippets found in knowledge base")
                return "I couldn't find any relevant information to answer your question."
            
            # Format the context
            logger.info("Formatting context from snippets")
            context = "\n\n".join([
                f"Source: {s['metadata']['source']}\n{s['content']}" 
                for s in snippets
            ])
            logger.info(f"Formatted context, length: {len(context)}")
            
            # Generate the answer using the RAG agent
            logger.info("Calling agent.run")
            prompt = f"""Answer the following question based on the provided context.
                If the context doesn't contain relevant information, say so.
                
                Question: {question}
                
                Context:
                {context}
                
                Answer:"""
            logger.info(f"Prompt length: {len(prompt)}")
            
            response = await self.agent.run(prompt)
            logger.info(f"Agent.run completed, response type: {type(response)}")
            
            # Handle AgentRunResult objects
            if hasattr(response, 'output') and str(type(response)) == "<class 'pydantic_ai.agent.AgentRunResult'>":
                logger.info("Converting AgentRunResult to string")
                return response.output
            # Handle other response types
            elif hasattr(response, 'text'):
                logger.info("Converting response.text to string")
                return response.text
            elif hasattr(response, 'content'):
                logger.info("Converting response.content to string")
                return response.content
            elif isinstance(response, str):
                logger.info(f"Response is string, length: {len(response)}")
                return response
            else:
                logger.info(f"Converting response to string, type: {type(response)}")
                return str(response)
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_msg
    
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
