"""Indexing module for creating FAISS embeddings of UN report chunks."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import faiss
from datetime import datetime
import os

# Embedding providers
import openai
from sentence_transformers import SentenceTransformer

from utils import load_config, ensure_dir, load_openai_key

logger = logging.getLogger(__name__)

class UNReportIndexer:
    """Creates and manages FAISS index for UN report chunks."""
    
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
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.embedding_provider}")
        
        if self.embedding_provider == 'openai':
            return self._embed_openai(texts, batch_size)
        else:
            return self._embed_local(texts, batch_size)
    
    def _embed_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        embeddings = []
        
        # Initialize client with API key
        client = openai.OpenAI(api_key=load_openai_key())
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.openai_model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"OpenAI embedding failed for batch {i//batch_size + 1}: {e}")
                raise
        
        return np.array(embeddings, dtype=np.float32)
    
    def _embed_local(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using local model."""
        try:
            # Process in batches to avoid memory issues
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.local_model.encode(batch, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            return np.vstack(embeddings).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise
    
    def create_index(self, chunks_file: Optional[str] = None) -> Dict[str, Any]:
        """Create FAISS index from parsed chunks."""
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
        
        logger.info(f"Loaded {len(chunks_df)} chunks")
        
        # Prepare texts for embedding
        texts = chunks_df['text'].tolist()
        
        # Generate embeddings
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
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search the index for similar chunks."""
        top_k = top_k or self.retrieval_config.get('top_k', 10)
        
        if self.index is None:
            logger.error("Index not loaded")
            return []
        
        logger.info(f"Searching for: '{query[:100]}...' (top_k={top_k})")
        
        try:
            # Generate query embedding
            query_embedding = self.embed_texts([query])
            
            if len(query_embedding) == 0:
                logger.error("Failed to generate query embedding")
                return []
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search with expanded results to allow for deduplication
            # We may need more chunks than top_k to get top_k unique documents
            search_k = min(top_k * 3, self.index.ntotal)  # Search 3x to account for duplicates
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Prepare results with deduplication by document
            results = []
            seen_documents = set()
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                
                chunk_data = self.chunk_metadata[idx].copy()
                
                # Use symbol as primary identifier, fallback to source_url, then title
                doc_id = chunk_data.get('symbol') or chunk_data.get('source_url') or chunk_data.get('title', f'chunk_{idx}')
                
                # Skip if we've already seen this document
                if doc_id in seen_documents:
                    continue
                
                seen_documents.add(doc_id)
                chunk_data['similarity_score'] = float(score)
                chunk_data['rank'] = len(results) + 1  # Rank based on unique documents, not chunks
                results.append(chunk_data)
                
                # Stop when we have enough unique documents
                if len(results) >= top_k:
                    break
            
            logger.info(f"Found {len(results)} unique documents from {len([s for s, i in zip(scores[0], indices[0]) if i != -1])} chunks")
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
        
        # Add metadata file info if available
        if Path(self.metadata_file).exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                stats['created_at'] = metadata.get('created_at')
            except:
                pass
        
        return stats
    
    def rebuild_if_needed(self, chunks_file: Optional[str] = None) -> Dict[str, Any]:
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
        return self.create_index(chunks_file)

def main():
    """Main function for standalone execution."""
    from utils import setup_logging
    
    setup_logging()
    config = load_config()
    
    indexer = UNReportIndexer(config)
    
    # Create or rebuild index
    result = indexer.rebuild_if_needed()
    
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