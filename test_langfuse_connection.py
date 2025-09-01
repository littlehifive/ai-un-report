#!/usr/bin/env python3
"""Test Langfuse connection and basic functionality."""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_langfuse_connection():
    """Test Langfuse connection and create a test trace."""
    
    # Load environment variables
    load_dotenv()
    
    try:
        from langfuse import Langfuse
        from langfuse_integration import langfuse_tracker
        
        print("🧪 Testing Langfuse Connection...")
        print("=" * 50)
        
        # Check environment variables
        public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
        secret_key = os.getenv('LANGFUSE_SECRET_KEY')
        host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        
        if not public_key or not secret_key:
            print("❌ Missing Langfuse API keys in .env file")
            print("Please update your .env file with:")
            print('LANGFUSE_PUBLIC_KEY="pk_lf_..."')
            print('LANGFUSE_SECRET_KEY="sk_lf_..."')
            return False
        
        print(f"✅ API keys found:")
        print(f"   Public Key: {public_key[:8]}...")
        print(f"   Secret Key: {secret_key[:8]}...")
        print(f"   Host: {host}")
        print()
        
        # Test direct Langfuse connection
        print("🔗 Testing direct Langfuse connection...")
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        
        # Test auth check
        auth_result = langfuse.auth_check()
        if auth_result:
            print("✅ Authentication successful")
        else:
            print("❌ Authentication failed")
            return False
        
        # Create a test trace using the newer API
        trace_id = langfuse.create_trace_id()
        
        print(f"✅ Test trace ID created: {trace_id}")
        
        # Create a test event
        langfuse.create_event(
            trace_id=trace_id,
            name="test_connection",
            input={"test": "connection"},
            output={"status": "success"},
            metadata={
                "test": True,
                "timestamp": datetime.now().isoformat(),
                "system": "un_reports_rag"
            }
        )
        
        print(f"✅ Test event created successfully")
        
        # Test our integration wrapper
        print()
        print("🔧 Testing UN RAG Langfuse integration...")
        
        if langfuse_tracker.enabled:
            print("✅ Langfuse tracker is enabled")
            
            # Test conversation tracking
            conversation_id = langfuse_tracker.start_conversation(
                session_id="test_session",
                user_id="test_user"
            )
            
            if conversation_id:
                print(f"✅ Test conversation created: {conversation_id}")
                
                # Test search tracking
                search_id = langfuse_tracker.track_search(
                    conversation_id,
                    "test query",
                    [{"symbol": "TEST_DOC", "similarity_score": 0.85}],
                    {"top_k": 5, "threshold": 0.3}
                )
                
                if search_id:
                    print(f"✅ Test search tracked: {search_id}")
                
                # Test generation tracking
                gen_id = langfuse_tracker.track_generation(
                    conversation_id,
                    "test query",
                    [{"symbol": "TEST_DOC", "text": "Test content"}],
                    "Test response from UN RAG system",
                    {"model": "gpt-4o-mini", "temperature": 0.1}
                )
                
                if gen_id:
                    print(f"✅ Test generation tracked: {gen_id}")
            
        else:
            print("❌ Langfuse tracker is not enabled")
            return False
        
        # Flush events
        langfuse.flush()
        langfuse_tracker.flush()
        
        print()
        print("🎉 SUCCESS! Langfuse is working correctly.")
        print()
        print("📊 View your test data at:")
        print(f"   {host}/project/[your-project-id]")
        print()
        print("🚀 Ready to run RAGAS evaluation with Langfuse tracking!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_langfuse_connection()
    sys.exit(0 if success else 1)