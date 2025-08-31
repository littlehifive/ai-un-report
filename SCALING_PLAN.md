# UN Reports RAG - Scaling Plan

## Current System Analysis

### ✅ What's Working
- OpenAI embeddings + FAISS (fast, accurate)
- Streamlit chat interface (user-friendly)
- Document parsing with PyMuPDF
- Basic chunking and metadata
- Cost-effective (~$5/year for 8K documents)

### ❌ Current Limitations
1. **Discovery**: Only 2/4 documents successfully downloaded
2. **Storage**: Files stored in repo (not scalable beyond ~100 docs)
3. **Reliability**: UN document URLs are unreliable 
4. **Scale**: No incremental updates, full rebuilds only
5. **Metadata**: Limited document classification
6. **Error Handling**: Poor resilience to failed downloads

---

## Recommended Approach: Phased Scaling

### Phase 1: Solid Foundation (100 Reports) 📚
**Goal**: Build robust system with 100 high-quality UN reports

**Key Changes:**
1. **Better Document Discovery**
   - Use UN ODS search API properly
   - Fallback to RSS feeds from major UN bodies
   - Manual curation of high-value report series
   
2. **Improved Storage Architecture**
   ```
   data/
   ├── documents/           # Raw PDFs (organized by year/body)
   │   ├── 2024/
   │   │   ├── general_assembly/
   │   │   ├── security_council/
   │   │   └── economic_social/
   │   └── 2025/
   ├── processed/           # Parsed chunks
   │   ├── chunks.parquet   # All text chunks
   │   └── metadata.parquet # Document metadata
   ├── index/               # Vector storage
   │   ├── faiss.index
   │   └── metadata.json
   └── cache/               # Download cache & logs
   ```

3. **Robust Download Pipeline**
   - Multiple URL attempts per document
   - Smart caching (skip re-downloads)
   - Progress tracking & resumable downloads
   - Better error handling & reporting

### Phase 2: Full Scale (1000+ Reports) 🚀
**Goal**: Handle full 2025 corpus efficiently

**Storage Strategy Decision:**
- **Keep files local** for 100-1000 documents (easier deployment)
- **Move to cloud storage** only if >2000 documents or multi-user

**Incremental Updates:**
- Daily/weekly incremental discovery
- Only re-index new/changed documents
- Version control for document updates

### Phase 3: Production Ready (All 2025) 🏭
**Goal**: Robust, scalable system for ongoing use

---

## Detailed Implementation Plan

### 1. Enhanced Document Discovery

```python
# Priority sources (in order):
1. UN ODS Direct Links (documents.un.org)
2. UNDL API (digitallibrary.un.org/api)  
3. RSS Feeds (reliablehigh-value sources)
4. Curated Lists (manual high-priority docs)
```

**Target Document Types:**
- Secretary-General Reports (A/78/*, A/79/*)
- Security Council Reports (S/2024/*, S/2025/*)
- General Assembly Resolutions (A/RES/78/*, A/RES/79/*)
- Economic & Social Council (E/2024/*, E/2025/*)
- Human Rights Council (A/HRC/*)

### 2. Storage Architecture Options

**Option A: Enhanced Local Storage (Recommended for Phase 1)**
```
Pros: Simple deployment, version control, no external deps
Cons: Repo size grows, not suitable for >1GB data
Best for: 100-500 documents
```

**Option B: Local + External Storage (Phase 2)**
```
Structure:
- Code repo: <100MB
- Data storage: External (AWS S3, Google Cloud, etc.)
- Local cache: Recent/frequently used documents

Pros: Unlimited scale, shared data, cost-effective
Cons: Deployment complexity, network dependency
Best for: 500+ documents, multiple users
```

**Option C: Database + File Storage (Phase 3)**
```
Database: PostgreSQL with pgvector for embeddings
File Storage: S3/GCS for PDFs
Local: FAISS index + metadata cache

Pros: Production-grade, real-time updates, multi-user
Cons: Infrastructure overhead
Best for: Production deployment
```

### 3. Recommended Starting Approach

**For Phase 1 (100 reports), stick with enhanced local storage:**

```bash
# Data structure:
data/
├── documents/              # ~50-100MB of PDFs
│   └── [organized by type/year]
├── processed/             # ~10-20MB parquet files  
├── index/                 # ~20-50MB FAISS + metadata
└── cache/                 # Logs, temp files
```

**Size estimates:**
- 100 documents ≈ 50-100MB
- Processed chunks ≈ 10-20MB  
- FAISS index ≈ 20-50MB
- **Total: ~100-200MB** (manageable in Git with LFS)

### 4. Implementation Roadmap

**Week 1: Foundation**
- [ ] Implement robust UN ODS document discovery
- [ ] Add multiple fallback URL strategies
- [ ] Create organized local storage structure
- [ ] Add download progress tracking & caching

**Week 2: Quality & Scale**
- [ ] Curate list of 100 high-priority 2025 reports
- [ ] Implement incremental discovery (new docs only)
- [ ] Add document metadata enrichment
- [ ] Improve error handling & reporting

**Week 3: Polish & Deploy**
- [ ] Add document categorization & filtering
- [ ] Improve chat UI with better search options
- [ ] Add corpus statistics dashboard
- [ ] Set up automated updates (daily/weekly)

### 5. Migration Strategy

**Step 1: Enhance Current System**
```bash
# Keep current structure, just improve it:
src/
├── discover.py     # Enhanced discovery with multiple sources
├── fetch.py        # Robust downloading with retries
├── parse.py        # Better metadata extraction
├── index.py        # Incremental indexing
└── app.py          # Enhanced UI
```

**Step 2: Scale Storage (when needed)**
```bash
# Only if >500 docs or multiple users:
- Add config option for external storage
- Keep local as default for simplicity
```

---

## Next Steps: Phase 1 Implementation

1. **Start with 100 carefully curated reports**
2. **Focus on reliability over quantity**  
3. **Use enhanced local storage** (simple & effective)
4. **Build robust discovery pipeline**
5. **Scale to 1000+ only after foundation is solid**

**Cost**: Phase 1 will cost ~$0.72 for indexing + <$2/year for personal use

**Timeline**: 2-3 weeks for a production-ready system with 100 reports

Would you like me to start implementing Phase 1 with the enhanced discovery and local storage approach?