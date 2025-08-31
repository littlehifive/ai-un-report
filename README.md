# UN Reports RAG System 🇺🇳

A production-ready RAG (Retrieval Augmented Generation) system for searching and chatting with UN reports from 2025. Built for easy forking, minimal setup, and cost-effective deployment.

## ✨ Features

- **150 High-Impact 2025 UN Reports** with strategic prioritization
- **Streamlit Chat Interface** with proper citations and links
- **Cost-Optimized Design** with local embedding fallbacks and rate limiting
- **One-Click Deployment** to Streamlit Community Cloud or Hugging Face Spaces
- **Fork-and-Run Architecture** - no database setup required
- **Respectful Crawling** with rate limiting and robots.txt compliance

## 🚀 Quick Start

```bash
# Clone and run
git clone your-repo
cd un-reports-rag
pip install -r requirements.txt

# Optional: Add OpenAI API key for better embeddings
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Start the app
streamlit run src/app.py
```

**That's it!** The system works out of the box with local embeddings or OpenAI.

## 📊 Current Corpus

The system includes **150 strategically selected 2025 UN reports**:

### Coverage by UN Body:
- **Secretary-General Reports**: 45 documents
- **General Assembly**: 38 documents  
- **Security Council**: 32 documents
- **Economic & Social Council**: 25 documents
- **UN System**: 10 documents

### Priority Topics:
- Climate change and sustainable development
- Peacekeeping and security operations  
- Human rights and humanitarian situations
- Annual progress reports
- High-impact policy documents

### Current Status ✅

The system is **production-ready** with:
- ✅ **150 UN reports** downloaded and indexed
- ✅ **965 content chunks** with proper metadata
- ✅ **OpenAI + local BGE** embedding support
- ✅ **Cost controls** and rate limiting active
- ✅ **Citations working** with correct UNDL links

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Discovery     │───▶│    Fetch     │───▶│   Parse     │───▶│    Index     │
│ (UN reports)    │    │ (PDF files)  │    │ (chunks)    │    │ (embeddings) │
└─────────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                      │
                                                                      ▼
                                                              ┌──────────────┐
                                                              │ Streamlit    │
                                                              │ Chat App     │
                                                              └──────────────┘
```

### Core Components

- **`discover.py`** - Finds recent UN reports using hybrid approach (RSS + sitemap + curated seeds)
- **`fetch.py`** - Downloads PDF/HTML files with manifest tracking and resume capability
- **`parse.py`** - Extracts text from documents, creates overlapping chunks with metadata
- **`index.py`** - Generates embeddings (OpenAI or local BGE) and builds FAISS index
- **`app.py`** - Streamlit chat interface with RAG pipeline and citation formatting

## Configuration

Edit `config.yaml` to customize:

```yaml
# Data collection
date_window_days: 365
languages: ["en"]
target_bodies: ["General Assembly", "Security Council", "Economic and Social Council"]

# Document processing  
parsing:
  chunk_tokens: 1200
  overlap_tokens: 150

# Retrieval settings
retrieval:
  top_k: 10
  embedding_provider: "openai"  # or "local_bge"
  
# OpenAI settings
openai:
  embedding_model: "text-embedding-3-small"
  chat_model: "gpt-4o-mini"
```

## Usage

### Command Line Tools

```bash
# Run individual pipeline steps
python src/discover.py  # Find reports
python src/fetch.py     # Download files  
python src/parse.py     # Extract & chunk text
python src/index.py     # Create embeddings index

# Full pipeline
make build

# Run evaluation tests
python src/eval.py

# Clean all generated data
make clean
```

### Streamlit App

```bash
make app  # or streamlit run src/app.py
```

Features:
- **Chat Interface**: Ask questions in natural language
- **Citations**: Every answer shows source UN documents with links
- **Corpus Management**: Rebuild index with latest reports
- **Filters**: Search by UN body, date range
- **Example Queries**: Pre-loaded sample questions

### Example Queries

- "What did the Secretary-General report on climate change?"
- "Recent Security Council resolutions on peacekeeping"
- "Economic and Social Council recommendations for development"
- "What are the main challenges mentioned in recent UN reports?"

## Data Sources & Compliance

- **Primary Source**: [UN Digital Library](https://digitallibrary.un.org) 
- **Robots.txt Compliant**: 5-second delays, respects disallowed paths
- **Discovery Method**: Hybrid approach using RSS feeds, sitemap crawling, and curated seeds
- **File Formats**: PDF (PyMuPDF), HTML (trafilatura)

## Deployment

### Streamlit Community Cloud

1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set `OPENAI_API_KEY` in secrets
4. Deploy `src/app.py`

### Hugging Face Spaces

1. Create new Space with Streamlit
2. Upload project files
3. Set `OPENAI_API_KEY` in settings
4. The app will auto-deploy

## Development

### Project Structure

```
ai-un-report/
├── src/
│   ├── discover.py     # UN reports discovery
│   ├── fetch.py        # File downloading
│   ├── parse.py        # Document parsing
│   ├── index.py        # FAISS indexing
│   ├── app.py          # Streamlit interface
│   ├── eval.py         # Evaluation tests
│   └── utils.py        # Shared utilities
├── data/
│   ├── raw/            # Downloaded files
│   ├── parsed/         # Processed chunks
│   └── *.faiss         # Vector index
├── scripts/
│   └── build_all.sh    # Pipeline script
├── config.yaml         # Configuration
├── requirements.txt    # Dependencies
└── Makefile           # Build commands
```

### Adding New Features

1. **New UN Bodies**: Add to `target_bodies` in `config.yaml`
2. **Different Languages**: Update `languages` list and modify parsing
3. **Custom Embeddings**: Implement new provider in `index.py`
4. **Enhanced Parsing**: Extend document structure detection in `parse.py`

### Testing

```bash
make test           # Basic import/config tests
python src/eval.py  # Full RAG evaluation with test queries
```

The evaluation runs 10 test queries and checks:
- Retrieval quality (relevant documents found)
- Generation quality (coherent answers with citations)
- Topic coverage (expected concepts mentioned)

## Technical Details

### Embedding Strategy
- **Primary**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Fallback**: Local BGE `BAAI/bge-small-en-v1.5` (384 dimensions)
- **Index**: FAISS IndexFlatIP with L2 normalization for cosine similarity

### Chunking Strategy
- **Size**: ~1200 tokens (4800 characters)
- **Overlap**: 150 tokens (600 characters)  
- **Boundaries**: Sentence and paragraph aware splitting
- **Metadata**: UN symbol, organ, date, section titles

### Rate Limiting
- **UN Digital Library**: 5-second delays (per robots.txt)
- **OpenAI API**: Batch processing with error handling
- **File Downloads**: Resumable with manifest tracking

## Limitations

- **Scope**: Last 365 days only (configurable)
- **Languages**: English focus (expandable)
- **Discovery**: Currently uses curated seed data + basic automation
- **File Types**: PDF and HTML only
- **Dependencies**: Requires OpenAI API for best performance

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Run `make test` and `make lint`
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **UN Digital Library** for providing public access to UN documents
- **OpenAI** for embedding and chat APIs
- **FAISS** for efficient similarity search
- **Streamlit** for rapid UI development
