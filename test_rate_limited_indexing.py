"""Test the rate-limited indexing system."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from utils import load_config, setup_logging
from index_improved import RateLimitedUNReportIndexer

def test_rate_limited_indexing():
    """Test the rate-limited indexing system."""
    
    print("🔍 Testing Rate-Limited Indexing System")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    config = load_config()
    
    # Step 1: Test with small batch
    print("\n1. Testing with small batch...")
    indexer = RateLimitedUNReportIndexer(config)
    
    # Create a small test dataset
    test_texts = [
        "This is a test document about UN reports.",
        "The Secretary-General submitted a report on climate change.",
        "The Security Council discussed peacekeeping operations.",
        "The General Assembly adopted a resolution on sustainable development.",
        "UNCTAD published its annual trade report.",
        "The Economic and Social Council reviewed progress on SDGs.",
        "Human rights violations were reported in the region.",
        "Peacekeeping forces were deployed to maintain stability."
    ]
    
    print(f"   Testing with {len(test_texts)} small texts...")
    
    try:
        embeddings = indexer.embed_texts(test_texts, batch_size=4)
        print(f"   ✅ Successfully generated embeddings: {embeddings.shape}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        
    except Exception as e:
        print(f"   ❌ Embedding generation failed: {e}")
        return False
    
    # Step 2: Test rate limiting logic
    print("\n2. Testing rate limiting logic...")
    
    # Test token estimation
    test_text = "This is a test text for token estimation."
    estimated_tokens = indexer._estimate_tokens(test_text)
    print(f"   Estimated tokens for '{test_text}': {estimated_tokens}")
    
    # Test rate limit checking
    can_make_request = indexer._check_rate_limits(1, 1000)
    print(f"   Can make request with 1000 tokens: {can_make_request}")
    
    # Step 3: Test with existing chunks (if available)
    print("\n3. Testing with existing chunks...")
    
    chunks_file = config['paths']['chunks_file']
    if Path(chunks_file).exists():
        try:
            chunks_df = pd.read_parquet(chunks_file)
            print(f"   Found {len(chunks_df)} chunks in {chunks_file}")
            
            # Limit to a small number for testing
            test_chunks = chunks_df.head(10)
            print(f"   Testing with {len(test_chunks)} chunks...")
            
            # Test embedding generation for these chunks
            test_texts = test_chunks['text'].tolist()
            embeddings = indexer.embed_texts(test_texts, batch_size=2)
            print(f"   ✅ Successfully generated embeddings for test chunks: {embeddings.shape}")
            
        except Exception as e:
            print(f"   ❌ Failed to process existing chunks: {e}")
    else:
        print(f"   No chunks file found at {chunks_file}")
    
    # Step 4: Test index creation with limited chunks
    print("\n4. Testing index creation...")
    
    if Path(chunks_file).exists():
        try:
            # Create index with very limited chunks to avoid rate limits
            result = indexer.create_index(chunks_file, max_chunks=5)
            
            if result['success']:
                print(f"   ✅ Index created successfully!")
                print(f"   Total chunks: {result['total_chunks']}")
                print(f"   Embedding dimension: {result['embedding_dim']}")
                print(f"   Provider: {result['embedding_provider']}")
                
                # Test search
                print("\n5. Testing search functionality...")
                test_query = "Secretary-General report"
                results = indexer.search(test_query, top_k=3)
                
                if results:
                    print(f"   ✅ Search successful! Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"      {i}. {result.get('title', 'No title')} (score: {result.get('similarity_score', 0):.3f})")
                else:
                    print(f"   ⚠️  No search results found")
                    
            else:
                print(f"   ❌ Index creation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"   ❌ Index creation failed: {e}")
    else:
        print(f"   ⚠️  No chunks file available for index testing")
    
    print("\n🎯 Rate-limited indexing test completed!")
    return True

def test_rate_limit_handling():
    """Test rate limit handling specifically."""
    
    print("\n🔄 Testing Rate Limit Handling...")
    
    setup_logging()
    config = load_config()
    indexer = RateLimitedUNReportIndexer(config)
    
    # Test rate limit counters
    print("   Testing rate limit counters...")
    
    # Simulate some requests
    indexer._update_rate_limit_counters(1000)
    indexer._update_rate_limit_counters(2000)
    
    print(f"   Request count: {indexer.request_count}")
    print(f"   Token count: {indexer.token_count}")
    
    # Test rate limit checking
    can_make_request = indexer._check_rate_limits(1, 500000)  # Large token request
    print(f"   Can make large request (500k tokens): {can_make_request}")
    
    can_make_request = indexer._check_rate_limits(1, 1000)  # Small token request
    print(f"   Can make small request (1k tokens): {can_make_request}")
    
    print("   ✅ Rate limit handling test completed!")

def test_configuration():
    """Test configuration settings."""
    
    print("\n⚙️  Testing Configuration...")
    
    setup_logging()
    config = load_config()
    
    # Check rate limiting settings
    rate_limit_config = config.get('rate_limiting', {})
    print(f"   Max queries per hour: {rate_limit_config.get('max_queries_per_hour', 'Not set')}")
    print(f"   Cooldown seconds: {rate_limit_config.get('cooldown_seconds', 'Not set')}")
    print(f"   Enable fallback: {rate_limit_config.get('enable_fallback', 'Not set')}")
    
    # Check embedding settings
    retrieval_config = config.get('retrieval', {})
    print(f"   Embedding provider: {retrieval_config.get('embedding_provider', 'Not set')}")
    print(f"   Fallback to local: {retrieval_config.get('fallback_to_local', 'Not set')}")
    
    # Check corpus settings
    corpus_config = config.get('corpus', {})
    print(f"   Target documents: {corpus_config.get('target_documents', 'Not set')}")
    print(f"   Core only: {corpus_config.get('core_only', 'Not set')}")
    
    print("   ✅ Configuration test completed!")

if __name__ == "__main__":
    print("Starting rate-limited indexing tests...")
    
    # Run tests
    success = test_rate_limited_indexing()
    
    if success:
        test_rate_limit_handling()
        test_configuration()
    
    print("\n🏁 All rate-limited indexing tests completed!")
