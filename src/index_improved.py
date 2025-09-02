"""Improved Indexing module with proper rate limiting for OpenAI API."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import faiss
from datetime import datetime
import os
import time
import math

# Embedding providers
import openai
from sentence_transformers import SentenceTransformer

from utils import load_config, ensure_dir, load_openai_key

logger = logging.getLogger(__name__)

class RateLimitedUNReportIndexer:
    """Creates and manages FAISS index with proper rate limiting for OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retrieval_config = config.get('retrieval', {})
        
        # Set up embedding provider
        self.embedding_provider = self.retrieval_config.get('embedding_provider', 'openai')
        self._setup_embeddings()
        
        # File paths
        self.index_file = config['paths']['index_file']
        self.metadata_file = config['paths']['metadata_file']
        self.chunks_file = config['paths']['chunks_file']
        
        # FAISS index
        self.index = None
        self.chunk_metadata = []
        self.embedding_dim = None
        
        # Rate limiting settings (read from config)
        self.rate_limit_config = config.get('rate_limiting', {})
        self.max_queries_per_hour = self.rate_limit_config.get('max_queries_per_hour', 20)
        self.cooldown_seconds = self.rate_limit_config.get('cooldown_seconds', 3)
        # OpenAI API rate limits
        self.openai_tpm_limit = self.rate_limit_config.get('openai_tpm_limit', 1_000_000)
        self.openai_rpm_limit = self.rate_limit_config.get('openai_rpm_limit', 3_500)
        # Default embedding batch size for OpenAI
        self.default_batch_size = self.rate_limit_config.get('batch_size', 8)

        self.last_request_time = 0
        self.request_count = 0
        self.token_count = 0
        self.reset_time = time.time() + 60  # Reset counter every minute
        
    def _setup_embeddings(self):
        """Initialize embedding model based on provider."""
        if self.embedding_provider == 'openai':
            api_key = load_openai_key()
            if not api_key:
                logger.warning("No OpenAI API key found, falling back to local embeddings")
                self.embedding_provider = 'local_bge'
            else:
                openai.api_key = api_key
                self.openai_model = self.config.get('openai', {}).get('embedding_model', 'text-embedding-3-small')
                logger.info(f"Using OpenAI embeddings: {self.openai_model}")
        
        if self.embedding_provider == 'local_bge':
            model_name = self.config.get('local_embeddings', {}).get('model', 'BAAI/bge-small-en-v1.5')
            device = self.config.get('local_embeddings', {}).get('device', 'cpu')
            
            logger.info(f"Loading local embedding model: {model_name}")
            self.local_model = SentenceTransformer(model_name, device=device)
            logger.info("Local embedding model loaded successfully")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _check_rate_limits(self, batch_size: int, estimated_tokens: int) -> bool:
        """Check if we can make a request without hitting rate limits."""
        current_time = time.time()
        
        # Reset counters if a minute has passed
        if current_time > self.reset_time:
            self.request_count = 0
            self.token_count = 0
            self.reset_time = current_time + 60
        
        # Check if we would exceed limits
        if (self.request_count + 1 > self.openai_rpm_limit or 
            self.token_count + estimated_tokens > self.openai_tpm_limit):
            return False
        
        return True
    
    def _wait_for_rate_limit(self, estimated_tokens: int):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        # Reset counters if a minute has passed
        if current_time > self.reset_time:
            self.request_count = 0
            self.token_count = 0
            self.reset_time = current_time + 60
        
        # Check if we need to wait
        while not self._check_rate_limits(1, estimated_tokens):
            wait_time = self.reset_time - current_time + 1  # Add 1 second buffer
            logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            current_time = time.time()
            
            # Reset counters after waiting
            if current_time > self.reset_time:
                self.request_count = 0
                self.token_count = 0
                self.reset_time = current_time + 60
    
    def _update_rate_limit_counters(self, tokens_used: int):
        """Update rate limit counters after a request."""
        self.request_count += 1
        self.token_count += tokens_used
    
    def embed_texts(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Generate embeddings for a list of texts with rate limiting."""
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.embedding_provider}")
        if batch_size is None:
            batch_size = self.default_batch_size
        
        if self.embedding_provider == 'openai':
            return self._embed_openai_rate_limited(texts, batch_size)
        else:
            return self._embed_local(texts, batch_size)
    
    def _embed_openai_rate_limited(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using OpenAI API with proper rate limiting."""
        embeddings = []
        client = openai.OpenAI(api_key=load_openai_key())
        
        total_batches = math.ceil(len(texts) / batch_size)
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Estimate tokens for this batch
            batch_tokens = sum(self._estimate_tokens(text) for text in batch)
            
            # Wait for rate limit if necessary
            self._wait_for_rate_limit(batch_tokens)
            
            try:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts, ~{batch_tokens} tokens)")
                
                response = client.embeddings.create(
                    input=batch,
                    model=self.openai_model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Update rate limit counters using response-level usage when available
                actual_tokens = None
                try:
                    usage = getattr(response, 'usage', None)
                    if usage is not None:
                        # openai v1 returns an object with total_tokens
                        if hasattr(usage, 'total_tokens'):
                            actual_tokens = usage.total_tokens
                        # in some cases usage may be a dict
                        elif isinstance(usage, dict) and 'total_tokens' in usage:
                            actual_tokens = usage['total_tokens']
                except Exception:
                    actual_tokens = None

                if actual_tokens is None:
                    # Fallback to our estimate for this batch
                    actual_tokens = batch_tokens

                self._update_rate_limit_counters(int(actual_tokens))
                
                logger.info(f"✅ Batch {batch_num}/{total_batches} completed successfully")
                
                # Add cooldown between batches
                if batch_num < total_batches:
                    time.sleep(self.cooldown_seconds)
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit for batch {batch_num}: {e}")
                # Wait longer and retry
                wait_time = 60  # Wait 1 minute
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
                # Reset counters
                self.request_count = 0
                self.token_count = 0
                self.reset_time = time.time() + 60
                
                # Retry this batch
                i -= batch_size  # Retry this batch
                continue
                
            except Exception as e:
                logger.error(f"OpenAI embedding failed for batch {batch_num}: {e}")
                raise
        
        return np.array(embeddings, dtype=np.float32)
    
    def _embed_local(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using local model."""
        try:
            # Process in batches to avoid memory issues
            embeddings = []
            total_batches = math.ceil(len(texts) / batch_size)
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                batch_embeddings = self.local_model.encode(batch, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
                
                logger.info(f"✅ Batch {batch_num}/{total_batches} completed")
            
            return np.vstack(embeddings).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise
    
    def create_index(self, chunks_file: Optional[str] = None, max_chunks: Optional[int] = None) -> Dict[str, Any]:
        """Create FAISS index from parsed chunks with optional chunk limit."""
        chunks_file = chunks_file or self.chunks_file
        
        logger.info(f"Creating index from {chunks_file}")
        
        # Load chunks
        try:
            chunks_df = pd.read_parquet(chunks_file)
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            return {'success': False, 'error': str(e)}
        
        if len(chunks_df) == 0:
            logger.error("No chunks found to index")
            return {'success': False, 'error': 'No chunks found'}
        
        # Limit chunks if specified
        if max_chunks and len(chunks_df) > max_chunks:
            logger.info(f"Limiting chunks from {len(chunks_df)} to {max_chunks}")
            chunks_df = chunks_df.head(max_chunks)
        
        logger.info(f"Processing {len(chunks_df)} chunks")
        
        # Prepare texts for embedding
        texts = chunks_df['text'].tolist()
        
        # Generate embeddings with rate limiting
        try:
            embeddings = self.embed_texts(texts)
            
            if len(embeddings) == 0:
                logger.error("No embeddings generated")
                return {'success': False, 'error': 'No embeddings generated'}
            
            self.embedding_dim = embeddings.shape[1]
            logger.info(f"Generated embeddings with dimension {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {'success': False, 'error': str(e)}
        
        # Create FAISS index
        try:
            # Use IndexFlatIP (Inner Product) for similarity search
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return {'success': False, 'error': str(e)}
        
        # Store chunk metadata
        self.chunk_metadata = chunks_df.to_dict('records')
        
        # Save index and metadata
        result = self._save_index()
        if not result['success']:
            return result
        
        summary = {
            'success': True,
            'total_chunks': len(chunks_df),
            'embedding_dim': self.embedding_dim,
            'embedding_provider': self.embedding_provider,
            'index_file': self.index_file,
            'metadata_file': self.metadata_file
        }
        
        logger.info(f"Index creation complete: {summary}")
        return summary
    
    def _save_index(self) -> Dict[str, Any]:
        """Save FAISS index and metadata to disk."""
        try:
            # Ensure directories exist
            ensure_dir(Path(self.index_file).parent)
            ensure_dir(Path(self.metadata_file).parent)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'embedding_provider': self.embedding_provider,
                'embedding_dim': self.embedding_dim,
                'total_chunks': len(self.chunk_metadata),
                'chunk_metadata': self.chunk_metadata
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved index to {self.index_file}")
            logger.info(f"Saved metadata to {self.metadata_file}")
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return {'success': False, 'error': str(e)}
    
    def load_index(self) -> Dict[str, Any]:
        """Load existing FAISS index and metadata."""
        logger.info("Loading existing index...")
        
        try:
            # Check if files exist
            if not Path(self.index_file).exists():
                return {'success': False, 'error': f'Index file not found: {self.index_file}'}
            
            if not Path(self.metadata_file).exists():
                return {'success': False, 'error': f'Metadata file not found: {self.metadata_file}'}
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_file)
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.embedding_dim = metadata['embedding_dim']
            self.chunk_metadata = metadata['chunk_metadata']
            
            # Verify embedding provider compatibility
            saved_provider = metadata.get('embedding_provider')
            if saved_provider != self.embedding_provider:
                logger.warning(f"Index was created with {saved_provider}, now using {self.embedding_provider}")
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
            return {
                'success': True,
                'total_chunks': len(self.chunk_metadata),
                'embedding_dim': self.embedding_dim,
                'created_at': metadata.get('created_at'),
                'embedding_provider': saved_provider
            }
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return {'success': False, 'error': str(e)}
    
    def search(self, query: str, top_k: int = None, *, min_threshold: float = None) -> List[Dict[str, Any]]:
        """Search the index for similar chunks."""
        top_k = top_k or self.retrieval_config.get('top_k', 10)
        min_threshold = min_threshold if min_threshold is not None else self.retrieval_config.get('min_similarity_threshold', 0.3)
        
        if self.index is None:
            logger.error("Index not loaded")
            return []
        
        logger.info(f"Searching for: '{query[:100]}...' (top_k={top_k}, threshold={min_threshold})")
        
        try:
            # Generate query embedding
            query_embedding = self.embed_texts([query])
            
            if len(query_embedding) == 0:
                logger.error("Failed to generate query embedding")
                return []
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search with expanded results to allow for deduplication
            search_k = min(top_k * 3, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Prepare results with deduplication by document and relevance filtering
            results = []
            seen_documents = set()
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:
                    continue
                
                if float(score) < min_threshold:
                    logger.debug(f"Skipping result with low similarity: {score:.3f} < {min_threshold}")
                    continue
                
                chunk_data = self.chunk_metadata[idx].copy()
                doc_id = chunk_data.get('symbol') or chunk_data.get('source_url') or chunk_data.get('title', f'chunk_{idx}')
                
                if doc_id in seen_documents:
                    continue
                
                seen_documents.add(doc_id)
                chunk_data['similarity_score'] = float(score)
                chunk_data['rank'] = len(results) + 1
                results.append(chunk_data)
                
                if len(results) >= top_k:
                    break
            
            total_chunks = len([s for s, i in zip(scores[0], indices[0]) if i != -1])
            relevant_chunks = len([s for s, i in zip(scores[0], indices[0]) if i != -1 and float(s) >= min_threshold])
            logger.info(f"Found {len(results)} relevant unique documents from {relevant_chunks}/{total_chunks} chunks above threshold {min_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if self.index is None:
            return {'loaded': False}
        
        stats = {
            'loaded': True,
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'embedding_provider': self.embedding_provider,
            'total_chunks': len(self.chunk_metadata)
        }
        
        if Path(self.metadata_file).exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                stats['created_at'] = metadata.get('created_at')
            except:
                pass
        
        return stats
    
    def rebuild_if_needed(self, chunks_file: Optional[str] = None, max_chunks: Optional[int] = None) -> Dict[str, Any]:
        """Rebuild index only if chunks file is newer than index."""
        chunks_file = chunks_file or self.chunks_file
        
        if not Path(chunks_file).exists():
            return {'success': False, 'error': 'Chunks file not found'}
        
        # Check if index exists and is newer than chunks
        if (Path(self.index_file).exists() and Path(self.metadata_file).exists()):
            chunks_mtime = Path(chunks_file).stat().st_mtime
            index_mtime = Path(self.index_file).stat().st_mtime
            
            if index_mtime > chunks_mtime:
                logger.info("Index is up to date, loading existing index")
                return self.load_index()
        
        logger.info("Index needs rebuilding")
        return self.create_index(chunks_file, max_chunks)


def main():
    """Main function for standalone execution."""
    from utils import setup_logging
    
    setup_logging()
    config = load_config()
    
    indexer = RateLimitedUNReportIndexer(config)
    
    # Create or rebuild index with chunk limit to avoid rate limits
    max_chunks = config.get('corpus', {}).get('max_chunks_total', 4000)  # Use new configurable limit
    result = indexer.rebuild_if_needed(max_chunks=max_chunks)
    
    if result['success']:
        print(f"Index ready!")
        print(f"Total chunks: {result['total_chunks']}")
        print(f"Embedding dimension: {result['embedding_dim']}")
        print(f"Provider: {result.get('embedding_provider', 'unknown')}")
        
        # Test search
        test_query = "Secretary-General report"
        results = indexer.search(test_query, top_k=3)
        
        if results:
            print(f"\nTest search for '{test_query}':")
            for i, result in enumerate(results[:3], 1):
                print(f"{i}. {result['title']} (score: {result['similarity_score']:.3f})")
        
    else:
        print(f"Index creation failed: {result.get('error')}")


if __name__ == "__main__":
    main()
