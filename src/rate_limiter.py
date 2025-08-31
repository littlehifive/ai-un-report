"""Rate limiting and cost control utilities."""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class QueryRateLimiter:
    """Rate limiter for user queries to control costs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('rate_limiting', {})
        self.max_queries_per_hour = self.config.get('max_queries_per_hour', 20)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 3)
        self.enable_fallback = self.config.get('enable_fallback', True)
        
        # Track queries in memory (for demo purposes)
        # In production, you'd use Redis or database
        self.query_log = []
        
    def can_query(self, user_id: str = "default") -> Dict[str, Any]:
        """Check if user can make a query."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old queries
        self.query_log = [q for q in self.query_log if q['timestamp'] > hour_ago]
        
        # Count queries in last hour
        user_queries = [q for q in self.query_log if q['user_id'] == user_id]
        
        if len(user_queries) >= self.max_queries_per_hour:
            return {
                'allowed': False,
                'reason': f'Rate limit exceeded: {len(user_queries)}/{self.max_queries_per_hour} queries per hour',
                'reset_time': max(q['timestamp'] for q in user_queries) + timedelta(hours=1),
                'suggest_fallback': self.enable_fallback
            }
        
        # Check cooldown
        if user_queries:
            last_query = max(q['timestamp'] for q in user_queries)
            cooldown_until = last_query + timedelta(seconds=self.cooldown_seconds)
            
            if now < cooldown_until:
                wait_seconds = (cooldown_until - now).total_seconds()
                return {
                    'allowed': False,
                    'reason': f'Cooldown active: wait {wait_seconds:.1f} seconds',
                    'wait_seconds': wait_seconds
                }
        
        return {'allowed': True, 'remaining': self.max_queries_per_hour - len(user_queries)}
    
    def record_query(self, user_id: str = "default", query_type: str = "search") -> None:
        """Record a query."""
        self.query_log.append({
            'user_id': user_id,
            'timestamp': datetime.now(),
            'type': query_type
        })
        logger.debug(f"Recorded query for {user_id}: {query_type}")


class CostOptimizer:
    """Optimize API calls and embeddings for cost control."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_config = config.get('openai', {})
        self.retrieval_config = config.get('retrieval', {})
        
    def should_use_local_embeddings(self) -> bool:
        """Determine if we should fall back to local embeddings."""
        import os
        
        # No OpenAI key = use local
        if not os.getenv('OPENAI_API_KEY'):
            logger.info("No OpenAI API key found, using local embeddings")
            return True
        
        # Explicit fallback enabled
        if self.retrieval_config.get('fallback_to_local', False):
            logger.info("Local embedding fallback enabled in config")
            return True
        
        return False
    
    def optimize_context(self, chunks: list, max_tokens: int = None) -> list:
        """Optimize context to fit within token limits."""
        if max_tokens is None:
            max_tokens = self.retrieval_config.get('max_context_tokens', 8000)
        
        # Rough estimation: 4 chars per token
        current_tokens = 0
        optimized_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            chunk_tokens = len(chunk_text) // 4
            
            if current_tokens + chunk_tokens <= max_tokens:
                optimized_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Truncate the last chunk if possible
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    remaining_chars = remaining_tokens * 4
                    truncated_chunk = chunk.copy()
                    truncated_chunk['text'] = chunk_text[:remaining_chars] + "..."
                    optimized_chunks.append(truncated_chunk)
                break
        
        if len(optimized_chunks) < len(chunks):
            logger.info(f"Context optimized: {len(chunks)} → {len(optimized_chunks)} chunks, ~{current_tokens} tokens")
        
        return optimized_chunks
    
    def get_cost_efficient_settings(self) -> Dict[str, Any]:
        """Get cost-optimized API settings."""
        return {
            'embedding_model': self.openai_config.get('embedding_model', 'text-embedding-3-small'),
            'chat_model': self.openai_config.get('chat_model', 'gpt-3.5-turbo'),
            'max_tokens': self.openai_config.get('max_tokens', 1000),
            'temperature': 0.1,  # Lower temperature for consistency and lower cost
            'use_local_embeddings': self.should_use_local_embeddings()
        }


def format_rate_limit_message(rate_info: Dict[str, Any]) -> str:
    """Format user-friendly rate limit message."""
    if rate_info.get('allowed', True):
        remaining = rate_info.get('remaining', 0)
        return f"✅ Query allowed ({remaining} remaining this hour)"
    
    reason = rate_info.get('reason', 'Unknown limit')
    
    if 'wait' in reason.lower():
        wait_time = rate_info.get('wait_seconds', 0)
        return f"⏳ Please wait {wait_time:.1f} seconds between queries"
    
    if 'rate limit' in reason.lower():
        reset_time = rate_info.get('reset_time')
        if reset_time:
            minutes_left = (reset_time - datetime.now()).total_seconds() / 60
            return f"🚫 Query limit reached. Resets in {minutes_left:.0f} minutes."
        
        if rate_info.get('suggest_fallback'):
            return f"🚫 {reason}\n💡 Try browsing documents manually or check back later."
    
    return f"🚫 {reason}"