"""Test script to debug the RAG pipeline."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
import pandas as pd
from utils import load_config, load_openai_key
from index import UNReportIndexer

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_rag_pipeline():
    """Test the RAG pipeline components."""
    
    print("=" * 60)
    print("Testing UN Reports RAG Pipeline")
    print("=" * 60)
    
    # Load config
    print("\n1. Loading configuration...")
    config = load_config()
    print(f"   Config loaded: {config.get('project', {}).get('name')}")
    
    # Check OpenAI key
    print("\n2. Checking OpenAI API key...")
    api_key = load_openai_key()
    if api_key:
        print(f"   ✓ API key found (length: {len(api_key)})")
    else:
        print("   ✗ API key NOT found")
        return
    
    # Check chunks file
    print("\n3. Checking chunks file...")
    chunks_file = Path(config['paths']['parsed_data']) / "chunks.parquet"
    if chunks_file.exists():
        df = pd.read_parquet(chunks_file)
        print(f"   ✓ Chunks file exists: {len(df)} chunks")
        print(f"   Columns: {df.columns.tolist()}")
        if len(df) > 0:
            print(f"   Sample chunk: {df.iloc[0]['text'][:100]}...")
    else:
        print(f"   ✗ Chunks file not found at {chunks_file}")
        return
    
    # Initialize indexer
    print("\n4. Initializing indexer...")
    indexer = UNReportIndexer(config)
    
    # Load index
    print("\n5. Loading index...")
    load_result = indexer.load_index()
    if load_result['success']:
        print(f"   ✓ Index loaded: {load_result['total_chunks']} chunks")
        print(f"   Embedding provider: {load_result.get('embedding_provider')}")
    else:
        print(f"   ✗ Failed to load index: {load_result.get('error')}")
        return
    
    # Test search
    print("\n6. Testing search functionality...")
    test_query = "What did the Secretary-General report on climate change?"
    print(f"   Query: {test_query}")
    
    try:
        results = indexer.search(test_query, top_k=5)
        print(f"   ✓ Search returned {len(results)} results")
        
        if results:
            print("\n   Top result:")
            result = results[0]
            print(f"   - Title: {result.get('title')}")
            print(f"   - Symbol: {result.get('symbol')}")
            print(f"   - Score: {result.get('similarity_score', 0):.3f}")
            print(f"   - Text preview: {result.get('text', '')[:200]}...")
        else:
            print("   ⚠ No results found")
            
    except Exception as e:
        print(f"   ✗ Search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_pipeline()