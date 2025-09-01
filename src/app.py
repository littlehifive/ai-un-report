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
from index_improved import RateLimitedUNReportIndexer as UNReportIndexer
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

def get_index_cache_key(config):
    """Generate cache key based on index file modification time."""
    index_path = Path(config['paths']['index_file'])
    meta_path = Path(config['paths']['index_file'].replace('.faiss', '.meta.json'))
    
    # Use modification time as cache key to auto-invalidate on index rebuild
    # Added version suffix to force cache invalidation after method signature change
    if index_path.exists() and meta_path.exists():
        return f"{index_path.stat().st_mtime}_{meta_path.stat().st_mtime}_v2"
    return "no_index_v2"

@st.cache_resource(show_spinner=False)
def initialize_indexer(_config, _cache_key):
    """Initialize and load the FAISS indexer (cached with auto-invalidation)."""
    indexer = UNReportIndexer(_config)
    load_result = indexer.load_index()
    
    if load_result['success']:
        st.success(f"✅ Index loaded: {load_result['total_chunks']} chunks from {load_result.get('index_date', 'unknown date')}")
        return indexer, load_result
    else:
        st.error(f"❌ Failed to load index: {load_result['error']}")
        return None, load_result

def get_conversation_context(messages: List[Dict[str, Any]], max_messages: int = 3) -> str:
    """Extract recent conversation context for better responses."""
    if len(messages) <= 1:  # Only current message
        return ""
    
    # Get last few message pairs (user + assistant)
    recent_messages = messages[-(max_messages * 2):]
    context_parts = []
    
    for msg in recent_messages:
        role = msg["role"]
        content = msg["content"][:300]  # Limit length
        if role == "user":
            context_parts.append(f"User previously asked: {content}")
        elif role == "assistant":
            context_parts.append(f"Assistant previously responded: {content}")
    
    return "\n".join(context_parts) if context_parts else ""

def extract_document_mentions(query: str) -> List[str]:
    """Extract specific UN document symbols or titles mentioned in query."""
    import re
    
    # Look for UN document symbols (A/79/123, S/2025/456, etc.)
    symbol_patterns = [
        r'\b[A-Z]+/\d+/\d+\b',      # A/79/123
        r'\b[A-Z]+/\d+\b',          # A/79
        r'\b[A-Z]/[A-Z\.]+/\d+\b',  # A/AC.109/2025
        r'\bS/\d+/\d+\b',           # S/2025/123
        r'\bE/\d+/\d+\b',           # E/2025/123
    ]
    
    mentions = []
    for pattern in symbol_patterns:
        matches = re.findall(pattern, query)
        mentions.extend(matches)
    
    # Look for document type mentions
    doc_keywords = ['this report', 'this document', 'the report', 'that document']
    for keyword in doc_keywords:
        if keyword.lower() in query.lower():
            mentions.append(keyword)
    
    return list(set(mentions))

def enhanced_search(indexer, query: str, conversation_history: List[Dict[str, Any]], 
                   top_k: int = 5, min_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Enhanced search that considers conversation context and document focus."""
    
    # Extract document mentions and conversation context
    document_mentions = extract_document_mentions(query)
    conv_context = get_conversation_context(conversation_history)
    
    # Get initial search results
    initial_results = indexer.search(query, top_k=top_k * 2, min_threshold=min_threshold)
    
    # If specific documents are mentioned, filter and prioritize them
    if document_mentions:
        logger.info(f"Document mentions detected: {document_mentions}")
        
        # Find chunks from mentioned documents
        focused_results = []
        other_results = []
        
        for result in initial_results:
            result_symbol = result.get('symbol', '')
            result_title = result.get('title', '').lower()
            
            # Check if this chunk is from a mentioned document
            is_mentioned = False
            for mention in document_mentions:
                if mention.lower() in ['this report', 'this document', 'the report', 'that document']:
                    # For generic references, check if previous conversation mentioned specific docs
                    if conv_context:
                        # Simple heuristic: if conversation context contains document symbols
                        import re
                        prev_symbols = re.findall(r'\b[A-Z]+/\d+(?:/\d+)?\b', conv_context)
                        if any(symbol in result_symbol for symbol in prev_symbols):
                            is_mentioned = True
                elif mention.upper() in result_symbol.upper():
                    is_mentioned = True
                elif mention.lower() in result_title:
                    is_mentioned = True
            
            if is_mentioned:
                focused_results.append(result)
            else:
                other_results.append(result)
        
        # Prioritize mentioned documents, then add others
        if focused_results:
            logger.info(f"Found {len(focused_results)} chunks from mentioned documents")
            # Take more from focused docs, fewer from others
            final_results = focused_results[:max(3, top_k-2)] + other_results[:2]
            return final_results[:top_k]
    
    # If no specific documents mentioned, use conversation context to enhance search
    if conv_context:
        # Extract key terms from conversation for context-aware search
        # This is a simple approach - in production you'd use more sophisticated NLP
        import re
        prev_terms = re.findall(r'\b(?:climate|sustainable|development|peacekeeping|security|economic|social|human rights|gender|humanitarian)\b', conv_context.lower())
        
        if prev_terms:
            logger.info(f"Conversation context terms: {set(prev_terms)}")
            # Boost results that match conversation themes
            themed_results = []
            other_results = []
            
            for result in initial_results:
                result_text = result.get('text', '').lower()
                if any(term in result_text for term in set(prev_terms)):
                    themed_results.append(result)
                else:
                    other_results.append(result)
            
            # Mix themed and other results
            final_results = []
            for i in range(top_k):
                if i < len(themed_results):
                    final_results.append(themed_results[i])
                elif (i - len(themed_results)) < len(other_results):
                    final_results.append(other_results[i - len(themed_results)])
            
            return final_results[:top_k]
    
    return initial_results[:top_k]

def get_chat_response(query: str, context_chunks: List[Dict[str, Any]], config: Dict[str, Any], 
                     conversation_history: List[Dict[str, Any]] = None) -> str:
    """Generate enhanced chat response with conversational awareness."""
    
    logger.info(f"get_chat_response called with query: {query}")
    logger.info(f"Number of context chunks: {len(context_chunks)}")
    
    # Check OpenAI API key
    api_key = load_openai_key()
    if not api_key:
        logger.error("OpenAI API key not found")
        return "❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
    
    logger.info("OpenAI API key found")
    
    # Extract conversation context
    conversation_history = conversation_history or []
    conv_context = get_conversation_context(conversation_history)
    document_mentions = extract_document_mentions(query)
    
    # Prepare context from chunks - use adaptive number based on relevance
    # Only include chunks that are actually relevant, up to a maximum of 5
    context_parts = []
    max_chunks_for_context = min(5, len(context_chunks))  # Use up to 5, but only if they exist
    
    for i, chunk in enumerate(context_chunks[:max_chunks_for_context], 1):
        # Optional: Add additional relevance check here if needed
        score = chunk.get('similarity_score', 0)
        logger.info(f"Including chunk {i} with similarity score: {score:.3f}")
        
        citation = f"[{i}] {chunk['title']} ({chunk['symbol']}, {chunk['date']})"
        context_parts.append(f"{citation}\n{chunk['text']}\n")
    
    context = "\n".join(context_parts)
    logger.info(f"Context prepared with {len(context_parts)} relevant chunks ({len(context)} characters)")
    
    # Build enhanced conversational prompt
    prompt_parts = []
    
    # System role and capabilities
    prompt_parts.append("""You are a knowledgeable UN analyst and conversational AI assistant specialized in United Nations reports and documents. You help users both search for information and have deeper analytical conversations about UN activities, policies, and findings.

CORE CAPABILITIES:
1. **Search Mode**: Help users find specific information across UN documents
2. **Analysis Mode**: Provide detailed analysis, synthesis, and insights from UN reports  
3. **Conversation Mode**: Maintain context and have meaningful discussions about UN topics

RESPONSE GUIDELINES:
- Always cite sources using [1], [2], etc. format
- Be conversational and engaging, not just factual
- Connect information across documents when relevant
- Acknowledge conversation history and build on previous exchanges
- If focusing on a specific document, prioritize information from that document
- Provide analysis and insights, not just facts
- Be precise but also contextual and explanatory""")

    # Add conversation context if available
    if conv_context:
        prompt_parts.append(f"\nCONVERSATION CONTEXT:\n{conv_context}")
    
    # Add document focus if mentioned
    if document_mentions:
        prompt_parts.append(f"\nUSER MENTIONED DOCUMENTS: {', '.join(document_mentions)}")
        prompt_parts.append("Focus primarily on these specific documents when answering.")
    
    # Add current context
    prompt_parts.append(f"\nCURRENT UN REPORT CONTEXT:\n{context}")
    
    # Add current query with instruction
    prompt_parts.append(f"\nCURRENT USER QUESTION: {query}")
    
    # Final instructions
    prompt_parts.append("""
RESPONSE APPROACH:
1. **Analyze**: Consider the conversation context and specific documents mentioned
2. **Synthesize**: Draw connections between different sources and previous discussion
3. **Respond**: Provide a thoughtful, well-cited response that advances the conversation

Remember: You're not just a search engine - you're a conversational partner helping users understand complex UN topics. Be helpful, insightful, and engaging while staying factual.""")
    
    prompt = "\n".join(prompt_parts)

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

def format_citations(chunks: List[Dict[str, Any]], citation_threshold: float = 0.4, response_text: str = "") -> str:
    """Format citations for display, filtering by relevance threshold."""
    if not chunks:
        return ""
    
    # Check if the response indicates no relevant information was found
    no_info_indicators = [
        "no specific information",
        "doesn't contain relevant information",
        "couldn't find any relevant",
        "no relevant information",
        "context doesn't contain",
        "unable to find specific"
    ]
    
    response_lower = response_text.lower()
    indicates_no_relevant_info = any(indicator in response_lower for indicator in no_info_indicators)
    
    # Filter chunks to only show those above citation threshold
    # This allows showing only highly relevant citations even if we used more chunks for context
    relevant_chunks = []
    for chunk in chunks:
        score = chunk.get('similarity_score', 0)
        if score >= citation_threshold:
            relevant_chunks.append(chunk)
    
    # If response indicates no relevant info, be more selective with citations
    if indicates_no_relevant_info:
        # Use a higher threshold when no relevant information was found
        higher_threshold = max(citation_threshold + 0.2, 0.6)
        very_relevant_chunks = [c for c in relevant_chunks if c.get('similarity_score', 0) >= higher_threshold]
        
        if not very_relevant_chunks:
            return "⚠️ **Note:** The search returned some results, but they don't appear to be directly relevant to your query. The similarity scores were too low to provide meaningful citations."
        
        # Show only 1-2 most relevant if response says no info
        relevant_chunks = very_relevant_chunks[:2]
        prefix = "⚠️ **Note:** Limited relevant sources found. These are the closest matches:\n\n"
    else:
        if not relevant_chunks:
            return "No sufficiently relevant sources found for this query."
        prefix = ""
    
    # Limit to maximum of 5 citations even if more are relevant
    max_citations = min(5, len(relevant_chunks))
    
    citations = []
    for i, chunk in enumerate(relevant_chunks[:max_citations], 1):
        title = chunk.get('title', 'Untitled')
        symbol = chunk.get('symbol', 'Unknown')
        date = chunk.get('date', 'Unknown date')
        organ = chunk.get('organ', 'Unknown organ')
        url = chunk.get('source_url', '#')
        score = chunk.get('similarity_score', 0)
        
        # Extract record ID for debugging
        record_id = url.split('/record/')[-1].split('/')[0] if '/record/' in url else 'unknown'
        
        citation = f"""
**[{i}]** [{title[:80]}...]({url})
- **UN Symbol:** {symbol}
- **Date:** {date}  
- **Organ:** {organ}
- **Relevance:** {score:.3f}
- **Record ID:** {record_id}
"""
        citations.append(citation)
    
    logger.info(f"Showing {len(citations)} citations from {len(chunks)} results (threshold: {citation_threshold}, no_info: {indicates_no_relevant_info})")
    return prefix + "\n".join(citations)

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
            # Limit chunks during rebuild to avoid hitting API limits
            max_docs = config.get('corpus', {}).get('target_documents', 500)
            max_chunks = max_docs * 2
            index_result = indexer.create_index(max_chunks=max_chunks)
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
    
    # Initialize indexer first (needed for both sidebar and main interface)
    cache_key = get_index_cache_key(config)
    indexer, load_result = initialize_indexer(config, cache_key)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Corpus Status")
        
        if indexer:
            stats = indexer.get_index_stats()
            
            # Show cache status for debugging
            st.caption(f"🔑 Cache key: `{cache_key[:20]}...`")
            
            # Calculate accurate document and chunk counts
            if hasattr(indexer, 'chunk_metadata') and indexer.chunk_metadata:
                # Count unique documents by symbol (most reliable identifier)
                doc_count = len(set(chunk.get('symbol', '') for chunk in indexer.chunk_metadata if chunk.get('symbol')))
                chunk_count = len(indexer.chunk_metadata)
                
                # File count removed to avoid confusion since 181 files != 52 documents
            else:
                doc_count = 0 
                chunk_count = 0
                
            st.metric("UN Documents", doc_count)
            st.metric("Content Chunks", chunk_count)
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
        
        if st.button("🧹 Clear Cache", help="Clear cached data and reload index"):
            st.cache_resource.clear()
            st.success("Cache cleared! The page will refresh...")
            st.rerun()
        
        if st.button("💬 New Conversation", help="Start a fresh conversation"):
            st.session_state.messages = []
            st.session_state.document_focus = None
            st.session_state.conversation_topic = None
            st.success("Started new conversation!")
            st.rerun()
        
        # Query settings
        st.header("⚙️ Search Settings")
        top_k = st.slider("Max results to retrieve", 1, 20, 5)
        min_threshold = st.slider("Minimum relevance threshold", 0.0, 1.0, 0.3, step=0.05, 
                                help="Lower values show more results, higher values show only highly relevant results")
        citation_threshold = st.slider("Citation relevance threshold", 0.0, 1.0, 0.4, step=0.05,
                                help="Only show citations with relevance above this threshold (higher = fewer but more relevant citations)")
        
        # Store settings in session state for access outside sidebar
        st.session_state.top_k = top_k
        st.session_state.min_threshold = min_threshold
        st.session_state.citation_threshold = citation_threshold
        
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
    
    # Initialize session state for conversation management
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "document_focus" not in st.session_state:
        st.session_state.document_focus = None
    
    if "conversation_topic" not in st.session_state:
        st.session_state.conversation_topic = None
    
    # Show conversation context if there is one
    if len(st.session_state.messages) > 0:
        if st.session_state.document_focus:
            st.info(f"📋 **Document Focus**: {st.session_state.document_focus}")
        elif st.session_state.conversation_topic:
            st.info(f"💭 **Topic**: {st.session_state.conversation_topic}")
    
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
                # Search for relevant chunks using enhanced search
                min_threshold = st.session_state.get('min_threshold', 0.3)
                logger.info(f"Searching with top_k={top_k}, threshold={min_threshold}")
                results = enhanced_search(indexer, query, st.session_state.messages, top_k=top_k, min_threshold=min_threshold)
                logger.info(f"Enhanced search returned {len(results)} results")
                
                if not results:
                    response = "I couldn't find any relevant information in the UN reports corpus for your query."
                    citations = ""
                    logger.info("No search results found")
                else:
                    # Generate response using retrieved context
                    logger.info("Generating chat response...")
                    response = get_chat_response(query, results, config, st.session_state.messages)
                    logger.info(f"Response generated: {response[:100]}...")
                    citation_threshold = st.session_state.get('citation_threshold', 0.4)
                    citations = format_citations(results, citation_threshold, response)
                    logger.info("Citations formatted")
                    
                    # Update conversation state based on results
                    if results:
                        # Check if user is focusing on a specific document
                        document_mentions = extract_document_mentions(query)
                        if document_mentions:
                            # Extract document symbols from results
                            doc_symbols = [r.get('symbol', '') for r in results[:2] if r.get('symbol')]
                            if doc_symbols:
                                st.session_state.document_focus = doc_symbols[0]
                        
                        # Extract topic from query for conversation continuity
                        topic_keywords = ['climate', 'sustainable development', 'peacekeeping', 'security', 
                                        'economic', 'social', 'human rights', 'gender', 'humanitarian']
                        for keyword in topic_keywords:
                            if keyword in query.lower():
                                st.session_state.conversation_topic = keyword.title()
                                break
                
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
        st.markdown("### 💡 Try these conversational examples:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Search Mode:**")
            search_queries = [
                "What recent reports discuss climate change?",
                "Show me Statistical Commission reports from 2025",
                "Find reports about sustainable development goals"
            ]
            
            for query in search_queries:
                if st.button(f"💬 {query}", key=f"search_{query[:15]}"):
                    # Add user message and process immediately
                    st.session_state.messages.append({"role": "user", "content": query})
                    
                    # Process the query immediately using the same logic as chat input
                    with st.spinner("Searching UN reports..."):
                        try:
                            min_threshold = st.session_state.get('min_threshold', 0.3)
                            top_k = st.session_state.get('top_k', 5)
                            results = enhanced_search(indexer, query, st.session_state.messages, top_k=top_k, min_threshold=min_threshold)
                            
                            if not results:
                                response = "I couldn't find any relevant information in the UN reports corpus for your query."
                                citations = ""
                            else:
                                response = get_chat_response(query, results, config, st.session_state.messages)
                                citation_threshold = st.session_state.get('citation_threshold', 0.4)
                                citations = format_citations(results, citation_threshold, response)
                                
                                # Update conversation state
                                document_mentions = extract_document_mentions(query)
                                if document_mentions and results:
                                    doc_symbols = [r.get('symbol', '') for r in results[:2] if r.get('symbol')]
                                    if doc_symbols:
                                        st.session_state.document_focus = doc_symbols[0]
                            
                            # Add assistant response
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "citations": citations
                            })
                            
                        except Exception as e:
                            logger.error(f"Error processing search query: {e}")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"❌ An error occurred: {str(e)}",
                                "citations": ""
                            })
                    
                    st.rerun()
        
        with col2:
            st.markdown("**💭 Conversation Mode:**")
            conversation_starters = [
                "Tell me about the latest technology and innovation report",
                "What are the main challenges in peacekeeping in recent reports?", 
                "Help me understand the economic situation discussed in UN reports"
            ]
            
            for query in conversation_starters:
                if st.button(f"💬 {query}", key=f"conv_{query[:15]}"):
                    # Add user message and process immediately
                    st.session_state.messages.append({"role": "user", "content": query})
                    
                    # Process the query immediately
                    with st.spinner("Searching UN reports..."):
                        try:
                            min_threshold = st.session_state.get('min_threshold', 0.3)
                            top_k = st.session_state.get('top_k', 5)
                            results = enhanced_search(indexer, query, st.session_state.messages, top_k=top_k, min_threshold=min_threshold)
                            
                            if not results:
                                response = "I couldn't find any relevant information in the UN reports corpus for your query."
                                citations = ""
                            else:
                                response = get_chat_response(query, results, config, st.session_state.messages)
                                citation_threshold = st.session_state.get('citation_threshold', 0.4)
                                citations = format_citations(results, citation_threshold, response)
                                
                                # Update conversation state
                                topic_keywords = ['climate', 'sustainable development', 'peacekeeping', 'security', 
                                                'economic', 'social', 'human rights', 'gender', 'humanitarian']
                                for keyword in topic_keywords:
                                    if keyword in query.lower():
                                        st.session_state.conversation_topic = keyword.title()
                                        break
                            
                            # Add assistant response
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "citations": citations
                            })
                            
                        except Exception as e:
                            logger.error(f"Error processing conversation query: {e}")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"❌ An error occurred: {str(e)}",
                                "citations": ""
                            })
                    
                    st.rerun()
        
        st.markdown("""
        **💡 Pro Tips:**
        - Ask follow-up questions like "What does this report say about..." 
        - Reference specific documents: "In report A/79/123, what..."
        - Build conversations: "How does this compare to previous years?"
        """)

if __name__ == "__main__":
    main()
