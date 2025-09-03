# AskUN - United Nations Report Assistant

A conversational AI assistant for exploring United Nations reports from 2025. Built with modern RAG (Retrieval Augmented Generation) technology to provide accurate, cited answers from official UN documents.

**Created by [Zezhen Wu](https://www.linkedin.com/in/zezhenwu/) • [GitHub Issues](https://github.com/littlehifive/ai-un-report)**

## 🚀 Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/littlehifive/ai-un-report
cd ai-un-report
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your OpenAI API key**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```
Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key-here
```

**4. Launch the application**
```bash
streamlit run src/app.py
```

That's it! The app will open in your browser and you can start chatting with UN documents immediately.

## ✨ Features

### 💬 **Conversational Interface**
- Natural language queries about UN reports and policies
- Follow-up questions with conversation memory
- Smart contextual search that understands references like "tell me more" or "what about climate change?"

### 🎯 **Accurate & Cited Responses**
- Every answer includes links to original UN documents
- Anti-hallucination system ensures responses come only from available documents
- Relevance scoring and citation validation prevent false information

### 🔍 **Advanced Search**
- Semantic search finds documents by meaning, not just keywords
- Filter by UN organs (Security Council, General Assembly, ECOSOC, etc.)
- Date range filtering for temporal analysis
- Real-time search through 500+ UN documents

### 📊 **Rich Document Coverage**
- Security Council reports and resolutions
- General Assembly proceedings and decisions  
- ECOSOC recommendations and analysis
- UNDP development reports
- Human Rights Council findings
- Secretary-General reports across all topics

## 🛠️ Technical Stack

- **Chat Model:** GPT-4o-mini (OpenAI) - Fast and accurate responses
- **Embeddings:** text-embedding-3-small (OpenAI) - Semantic search
- **Vector Database:** FAISS (local) - No external dependencies
- **Frontend:** Streamlit - Clean, responsive chat interface
- **Evaluation:** LangFuse monitoring - Quality assurance
- **Data Source:** [UN Digital Library](https://digitallibrary.un.org/?ln=en) via [Record API](https://digitallibrary.un.org/help/record-api?ln=en)

## 📁 Project Structure

```
ai-un-report/
├── src/
│   ├── app.py                  # 🌐 Main Streamlit application
│   ├── indexer.py              # 🔍 Vector indexing & search
│   ├── utils.py                # 🛠 Configuration & utilities
│   ├── discover.py             # 📋 UN document discovery (corpus rebuilding)
│   ├── fetch.py                # 📥 Document downloading (corpus rebuilding)  
│   └── parse.py                # 📄 Text extraction (corpus rebuilding)
├── data/
│   ├── index.faiss            # Vector search index
│   ├── index.meta.json        # Index metadata
│   ├── records.parquet        # Document metadata
│   ├── parsed/chunks.parquet  # Processed text chunks
│   └── 2025-core/raw/         # UN PDF documents (500+ files)
├── scripts/build_all.sh       # 🔄 Full corpus rebuild pipeline
├── config.yaml               # ⚙️ System configuration
├── requirements.txt          # 📦 Python dependencies
└── README.md                 # 📖 This documentation
```

## ⚙️ Configuration

The system works out-of-the-box, but you can customize settings in `config.yaml`:

```yaml
# OpenAI API settings
openai:
  chat_model: "gpt-4o-mini"        # Fast, accurate model
  embedding_model: "text-embedding-3-small"
  max_tokens: 2000

# Search behavior
search:
  top_k: 5                        # Documents per search
  min_threshold: 0.3              # Relevance threshold

# Corpus management
corpus:
  target_documents: 500           # Total docs to index (for API cost control)
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run src/app.py
```

### Streamlit Cloud (Recommended)
1. Fork this repository on GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Add `OPENAI_API_KEY` to Streamlit secrets
4. Deploy automatically

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.address", "0.0.0.0"]
```

## 🔄 Updating the Document Corpus

The system comes pre-built with 500+ UN documents from 2025. To refresh with newer documents:

```bash
# Full rebuild (discover → fetch → parse → index)
bash scripts/build_all.sh
```

**⚠️ Note:** Rebuilding requires significant OpenAI API usage for re-embedding all documents. The pre-built corpus is sufficient for most users.

## 💻 System Requirements

- **Python:** 3.11+ (required for modern ML libraries)
- **Memory:** 2GB+ RAM (for FAISS index loading)
- **Storage:** 1GB+ (for UN PDF documents and search index)
- **API:** OpenAI API key with GPT-4 access

## 📈 Performance

- **Response Time:** <10 seconds average
- **Document Coverage:** 500+ unique UN reports from 2025
- **Search Accuracy:** Optimized for relevance and citation quality
- **Uptime:** Designed for 24/7 operation on Streamlit Cloud

## 🎯 Example Queries

Try these conversational examples:

**📋 Document-Specific:**
- "What does the UNCTAD Technology and Innovation Report 2025 say about AI?"
- "Find the Secretary-General's recommendations on sustainable development"
- "What peacekeeping challenges are mentioned in Security Council reports?"

**🗣️ Conversational:**
- "How do UN reports address climate change?" → "Tell me more about adaptation strategies"
- "What humanitarian crises are highlighted?" → "What about funding gaps?"
- "Show me gender equality initiatives" → "How do these relate to SDGs?"

**🔍 Analytical:**
- "Compare different UN agencies' approaches to economic development"
- "What organizational reforms does the Joint Inspection Unit recommend?"
- "How has UN peacekeeping strategy evolved in 2025 reports?"

## 📊 Data & Transparency

- **Source:** All documents sourced from the official [UN Digital Library](https://digitallibrary.un.org/?ln=en)
- **Processing:** Transparent RAG pipeline with full source attribution
- **Quality:** Every response includes direct links to original UN documents
- **Coverage:** Comprehensive across all major UN bodies and specialized agencies

## 🤝 Contributing

This project was built collaboratively with Claude Code and is ready for community contributions.

**For issues or suggestions:**
1. [Submit an issue](https://github.com/littlehifive/ai-un-report/issues) on GitHub
2. Test thoroughly with representative UN document queries
3. Ensure citation validation remains intact in any changes

**Areas for contribution:**
- Additional UN document sources
- Enhanced conversation flows
- Performance optimizations
- Deployment guides for other platforms

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **UN Digital Library:** For providing comprehensive access to UN documents
- **OpenAI:** For GPT-4o-mini and embedding models
- **Streamlit:** For the excellent web application framework
- **Claude Code:** For collaborative development support

---

**Ready to explore UN reports with AI?** 🚀 

[Deploy on Streamlit Cloud](https://share.streamlit.io) • [Follow @zezhenwu](https://www.linkedin.com/in/zezhenwu/) • [Star on GitHub](https://github.com/littlehifive/ai-un-report)