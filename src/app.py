"""Streamlit chat interface for UN Reports RAG system."""

import streamlit as st
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import openai
import json

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent))

from utils import load_config, load_openai_key, setup_logging
from index import UNReportIndexer
from discover import UNReportDiscoverer
from fetch import UNReportFetcher
from parse import UNReportParser

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="UN Reports RAG Chat",
    page_icon="🇺🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_app_config():
    """Load configuration (cached)."""
    return load_config()

@st.cache_resource
def initialize_indexer(_config):
    """Initialize and load the FAISS indexer (cached)."""
    indexer = UNReportIndexer(_config)
    load_result = indexer.load_index()
    
    if load_result['success']:
        st.success(f"Index loaded: {load_result['total_chunks']} chunks")
        return indexer, load_result
    else:
        st.error(f"Failed to load index: {load_result['error']}")
        return None, load_result

def get_chat_response(query: str, context_chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    """Generate chat response using OpenAI with retrieved context."""
    
    logger.info(f"get_chat_response called with query: {query}")
    logger.info(f"Number of context chunks: {len(context_chunks)}")
    
    # Check OpenAI API key
    api_key = load_openai_key()
    if not api_key:
        logger.error("OpenAI API key not found")
        return "❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    logger.info("OpenAI API key found")
    
    # Prepare context from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks[:5], 1):  # Limit to top 5 chunks
        citation = f"[{i}] {chunk['title']} ({chunk['symbol']}, {chunk['date']})"
        context_parts.append(f"{citation}\n{chunk['text']}\n")
    
    context = "\n".join(context_parts)
    logger.info(f"Context prepared with {len(context)} characters")
    
    # Create prompt
    prompt = f"""You are an AI assistant that helps users understand UN reports. Use the provided context to answer the user's question accurately and concisely.

IMPORTANT INSTRUCTIONS:
- Answer based ONLY on the provided context from UN reports
- Always cite your sources using the format [1], [2], etc.
- If the context doesn't contain relevant information, say so clearly
- Be precise and factual
- Do not make up or hallucinate information

CONTEXT FROM UN REPORTS:
{context}

USER QUESTION: {query}

ANSWER:"""

    logger.info(f"Prompt length: {len(prompt)} characters")

    try:
        logger.info("Creating OpenAI client...")
        client = openai.OpenAI(api_key=api_key)
        
        model_name = config.get('openai', {}).get('chat_model', 'gpt-3.5-turbo')
        max_tokens = config.get('openai', {}).get('max_tokens', 2000)
        
        logger.info(f"Calling OpenAI API with model: {model_name}, max_tokens: {max_tokens}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        logger.info(f"OpenAI response received, length: {len(answer)} characters")
        
        return answer
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"❌ Error generating response: {str(e)}"

def format_citations(chunks: List[Dict[str, Any]]) -> str:
    """Format citations for display."""
    if not chunks:
        return ""
    
    citations = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get('title', 'Untitled')
        symbol = chunk.get('symbol', 'Unknown')
        date = chunk.get('date', 'Unknown date')
        organ = chunk.get('organ', 'Unknown organ')
        url = chunk.get('source_url', '#')
        score = chunk.get('similarity_score', 0)
        
        citation = f"""
**[{i}]** [{title}]({url})
- **UN Symbol:** {symbol}
- **Date:** {date}  
- **Organ:** {organ}
- **Relevance:** {score:.3f}
"""
        citations.append(citation)
    
    return "\n".join(citations)

def rebuild_corpus():
    """Rebuild the entire corpus (discover -> fetch -> parse -> index)."""
    config = load_app_config()
    
    with st.spinner("Rebuilding corpus..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Discovery
            status_text.text("🔍 Discovering UN reports...")
            discoverer = UNReportDiscoverer(config)
            reports = discoverer.discover_all()
            discoverer.save_records(reports, config['paths']['records_file'])
            progress_bar.progress(25)
            
            # Step 2: Fetch
            status_text.text("📥 Downloading report files...")
            fetcher = UNReportFetcher(config)
            fetch_result = fetcher.fetch_from_records(config['paths']['records_file'])
            progress_bar.progress(50)
            
            if not fetch_result['success']:
                st.error(f"Fetch failed: {fetch_result.get('error')}")
                return False
            
            # Step 3: Parse
            status_text.text("📄 Parsing documents...")
            parser = UNReportParser(config)
            manifest_file = Path(config['paths']['raw_data']) / "files_manifest.parquet"
            parse_result = parser.parse_all_files(config['paths']['records_file'], str(manifest_file))
            progress_bar.progress(75)
            
            if not parse_result['success']:
                st.error(f"Parsing failed: {parse_result.get('error')}")
                return False
            
            # Step 4: Index
            status_text.text("🔍 Creating embeddings index...")
            indexer = UNReportIndexer(config)
            index_result = indexer.create_index()
            progress_bar.progress(100)
            
            if index_result['success']:
                status_text.text("✅ Corpus rebuild complete!")
                st.success(f"""
Corpus rebuilt successfully!
- **Reports discovered:** {len(reports)}
- **Files downloaded:** {fetch_result['successful_downloads']}
- **Chunks created:** {parse_result['total_chunks']}
- **Index vectors:** {index_result['total_chunks']}
                """)
                return True
            else:
                st.error(f"Indexing failed: {index_result.get('error')}")
                return False
                
        except Exception as e:
            st.error(f"Rebuild failed: {str(e)}")
            return False

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🇺🇳 UN Reports RAG Chat")
    st.markdown("Ask questions about recent UN reports and get answers with citations.")
    
    # Load configuration
    config = load_app_config()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Corpus Status")
        
        # Try to load indexer
        indexer, load_result = initialize_indexer(config)
        
        if indexer:
            stats = indexer.get_index_stats()
            st.metric("Total Documents", stats.get('total_chunks', 0))
            st.metric("Embedding Provider", stats.get('embedding_provider', 'Unknown'))
            
            if stats.get('created_at'):
                created_date = datetime.fromisoformat(stats['created_at'].replace('Z', '+00:00'))
                st.metric("Last Updated", created_date.strftime("%Y-%m-%d"))
        else:
            st.warning("Index not available")
        
        st.markdown("---")
        
        # Corpus controls
        st.header("🔧 Controls")
        
        if st.button("🔄 Rebuild Corpus", help="Discover, fetch, parse, and index recent UN reports"):
            if rebuild_corpus():
                st.rerun()  # Refresh the app after rebuild
        
        # Query settings
        st.header("⚙️ Search Settings")
        top_k = st.slider("Results to retrieve", 1, 20, 5)
        
        # Store top_k in session state for access outside sidebar
        st.session_state.top_k = top_k
        
        # Filters (if we have data)
        if indexer and indexer.chunk_metadata:
            st.header("🔍 Filters")
            
            # Get available organs
            all_organs = list(set(chunk.get('organ', '') for chunk in indexer.chunk_metadata if chunk.get('organ')))
            selected_organs = st.multiselect("UN Bodies", all_organs, default=all_organs[:3])
            
            # Date range
            st.subheader("Date Range")
            date_filter = st.checkbox("Filter by date", value=False)
            if date_filter:
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("From", value=datetime.now() - timedelta(days=365))
                with col2:
                    end_date = st.date_input("To", value=datetime.now())
    
    # Main chat interface
    if not indexer:
        st.error("❌ Search index not available. Please rebuild the corpus using the sidebar.")
        st.info("💡 Click 'Rebuild Corpus' in the sidebar to download and index recent UN reports.")
        return
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show citations if available
            if message.get("citations"):
                with st.expander("📚 Sources"):
                    st.markdown(message["citations"])
    
    # Chat input
    if query := st.chat_input("Ask about UN reports..."):
        logger.info(f"User query received: {query}")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.spinner("Searching UN reports..."):
            try:
                # Search for relevant chunks
                logger.info(f"Searching with top_k={top_k}")
                results = indexer.search(query, top_k=top_k)
                logger.info(f"Search returned {len(results)} results")
                
                if not results:
                    response = "I couldn't find any relevant information in the UN reports corpus for your query."
                    citations = ""
                    logger.info("No search results found")
                else:
                    # Generate response using retrieved context
                    logger.info("Generating chat response...")
                    response = get_chat_response(query, results, config)
                    logger.info(f"Response generated: {response[:100]}...")
                    citations = format_citations(results)
                    logger.info("Citations formatted")
                
                # Display response in chat message
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
                    # Show citations
                    if citations:
                        with st.expander("📚 Sources"):
                            st.markdown(citations)
                
                # Add to chat history AFTER displaying
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "citations": citations
                })
                logger.info("Response added to chat history")
                
            except Exception as e:
                logger.error(f"Error in chat flow: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                with st.chat_message("assistant"):
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.error("Check the terminal logs for details.")
    
    # Example queries
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Try asking about:")
        example_queries = [
            "What did the Secretary-General report on climate change?",
            "Recent Security Council resolutions on peacekeeping",
            "Economic and Social Council recommendations for sustainable development",
            "What are the main challenges mentioned in recent UN reports?"
        ]
        
        for example_query in example_queries:
            if st.button(f"💬 {example_query}", key=f"example_{example_query[:20]}"):
                # Process the query immediately
                logger.info(f"Processing example query: {example_query}")
                
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": example_query})
                
                # Generate response
                with st.spinner("Searching UN reports..."):
                    try:
                        # Search for relevant chunks
                        search_top_k = st.session_state.get('top_k', 5)
                        logger.info(f"Searching with top_k={search_top_k}")
                        results = indexer.search(example_query, top_k=search_top_k)
                        logger.info(f"Search returned {len(results)} results")
                        
                        if not results:
                            response = "I couldn't find any relevant information in the UN reports corpus for your query."
                            citations = ""
                        else:
                            # Generate response using retrieved context
                            logger.info("Generating chat response...")
                            response = get_chat_response(example_query, results, config)
                            logger.info(f"Response generated: {response[:100]}...")
                            citations = format_citations(results)
                        
                        # Add assistant response to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "citations": citations
                        })
                        
                        # Rerun to display the messages
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Error processing example query: {e}")
                        st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()