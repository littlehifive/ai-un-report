"""Langfuse integration for UN Reports RAG system monitoring."""

import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

# Langfuse imports
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    print("Langfuse not installed. Run: pip install langfuse")
    Langfuse = None

logger = logging.getLogger(__name__)

class UNRAGLangfuseTracker:
    """Langfuse tracking integration for UN Reports RAG."""
    
    def __init__(self):
        """Initialize Langfuse client."""
        self.langfuse = None
        self.enabled = False
        
        # Check for Langfuse credentials
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY") 
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if public_key and secret_key and Langfuse:
            try:
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                self.enabled = True
                logger.info("Langfuse integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")
        else:
            logger.info("Langfuse disabled (missing credentials or library)")
    
    def start_conversation(self, session_id: str, user_id: Optional[str] = None) -> Optional[str]:
        """Start tracking a conversation session."""
        if not self.enabled:
            return None
        
        try:
            trace_id = self.langfuse.create_trace_id()
            
            # Create initial event for the conversation
            self.langfuse.create_event(
                trace_id=trace_id,
                name="un_rag_conversation_start",
                input={"session_id": session_id, "user_id": user_id},
                metadata={
                    "system": "un_reports_rag",
                    "start_time": datetime.now().isoformat()
                }
            )
            return trace_id
        except Exception as e:
            logger.error(f"Failed to start conversation tracking: {e}")
            return None
    
    def track_search(self, 
                    trace_id: str,
                    query: str, 
                    results: List[Dict[str, Any]], 
                    search_params: Dict[str, Any]) -> Optional[str]:
        """Track search/retrieval operations."""
        if not self.enabled or not trace_id:
            return None
        
        try:
            self.langfuse.create_event(
                trace_id=trace_id,
                name="enhanced_search",
                input={
                    "query": query,
                    "search_params": search_params
                },
                output={
                    "num_results": len(results),
                    "result_scores": [r.get('similarity_score', 0) for r in results[:3]],
                    "result_symbols": [r.get('symbol', '') for r in results[:3]]
                },
                metadata={
                    "search_type": "enhanced_semantic_search",
                    "index_size": search_params.get("index_size", "unknown")
                }
            )
            return trace_id
        except Exception as e:
            logger.error(f"Failed to track search: {e}")
            return None
    
    def track_generation(self, 
                        trace_id: str,
                        query: str,
                        context_chunks: List[Dict[str, Any]],
                        response: str,
                        model_params: Dict[str, Any],
                        conversation_history: List[Dict[str, Any]] = None) -> Optional[str]:
        """Track response generation."""
        if not self.enabled or not trace_id:
            return None
        
        try:
            # Calculate token usage estimation
            input_tokens = len(query.split()) + sum(len(chunk.get('text', '').split()) for chunk in context_chunks)
            output_tokens = len(response.split())
            
            self.langfuse.create_event(
                trace_id=trace_id,
                name="un_rag_generation",
                input={
                    "query": query,
                    "context_chunks": len(context_chunks),
                    "conversation_turns": len(conversation_history or [])
                },
                output=response,
                metadata={
                    "model": model_params.get("model", "unknown"),
                    "system_prompt_type": "conversational_rag",
                    "context_sources": [chunk.get('symbol', '') for chunk in context_chunks],
                    "temperature": model_params.get("temperature", 0.1),
                    "max_tokens": model_params.get("max_tokens", 1000),
                    "usage": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                        "unit": "TOKENS"
                    }
                }
            )
            return trace_id
        except Exception as e:
            logger.error(f"Failed to track generation: {e}")
            return None
    
    def track_user_feedback(self, 
                           trace_id: str,
                           score: float,
                           comment: str = None,
                           feedback_type: str = "thumbs") -> bool:
        """Track user feedback on responses."""
        if not self.enabled or not trace_id:
            return False
        
        try:
            self.langfuse.score(
                trace_id=trace_id,
                name=feedback_type,
                value=score,
                comment=comment
            )
            return True
        except Exception as e:
            logger.error(f"Failed to track feedback: {e}")
            return False
    
    def track_error(self, 
                   trace_id: str,
                   error: Exception,
                   context: Dict[str, Any] = None) -> bool:
        """Track errors during RAG operations."""
        if not self.enabled or not trace_id:
            return False
        
        try:
            self.langfuse.event(
                trace_id=trace_id,
                name="rag_error",
                input=context or {},
                output={
                    "error_type": type(error).__name__,
                    "error_message": str(error)
                },
                level="ERROR"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to track error: {e}")
            return False
    
    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse events: {e}")

# Global tracker instance
langfuse_tracker = UNRAGLangfuseTracker()

# Decorator for easy tracking
def track_rag_operation(operation_name: str):
    """Decorator to track RAG operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not langfuse_tracker.enabled:
                return func(*args, **kwargs)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful operation
                langfuse_tracker.langfuse.event(
                    name=f"rag_{operation_name}",
                    input={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                    output={"success": True, "duration_ms": duration * 1000},
                    level="INFO"
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log failed operation
                langfuse_tracker.langfuse.event(
                    name=f"rag_{operation_name}_error",
                    input={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                    output={
                        "success": False, 
                        "error": str(e), 
                        "duration_ms": duration * 1000
                    },
                    level="ERROR"
                )
                raise
        
        return wrapper
    return decorator