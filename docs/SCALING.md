# Scaling UN Reports RAG System

This document explains how to scale beyond the demo corpus to handle larger datasets and production workloads.

## Current Demo Setup

**Target Audience:** Developers, researchers, organizations wanting a quick UN reports search  
**Corpus Size:** ~150 high-impact 2025 UN reports  
**Storage:** Local files (GitHub-friendly, <2GB)  
**Deployment:** Streamlit Community Cloud, Hugging Face Spaces  
**Cost:** Minimal (OpenAI API usage only)  

## Scaling Options

### Option 1: Extended Local Corpus (2-5GB)

**Good for:** Organizations with specific focus areas wanting more documents

```yaml
# config.yaml modifications
corpus:
  target_documents: 500-1000    # Expand to 500-1000 documents
  max_size_gb: 5               # Allow larger repo
  include_archives: true       # Add 2024, 2023 data
  extended_search: true        # More comprehensive discovery
```

**Implementation:**
1. Modify discovery queries in `discover_enhanced.py`
2. Add year-based corpus management
3. Use Git LFS for larger files
4. Deploy to dedicated servers (not free tiers)

### Option 2: Database Backend (Production Scale)

**Good for:** Full UN document corpus, multiple organizations, public websites

#### Database Setup

**Recommended Stack:**
- **Database:** PostgreSQL 15+ with pgvector extension
- **Embeddings:** Store vectors in database
- **Files:** Object storage (S3, CloudFlare R2)
- **Search:** Hybrid search (vector + full-text)

```sql
-- Database schema
CREATE EXTENSION vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(100) UNIQUE,
    title TEXT,
    date DATE,
    organ VARCHAR(100),
    language VARCHAR(10),
    file_url TEXT,
    record_url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    chunk_text TEXT,
    embedding vector(1536),
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_documents_date ON documents(date);
CREATE INDEX idx_documents_organ ON documents(organ);
```

#### Migration Steps

1. **Setup Database**
```bash
# Using Docker
docker run -d \
  --name un-postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -p 5432:5432 \
  pgvector/pgvector:pg15
```

2. **Update Configuration**
```python
# config.yaml
database:
  enabled: true
  url: "postgresql://user:pass@localhost:5432/unrag"
  connection_pool: 20
  
storage:
  type: "s3"  # or "local" for transition
  bucket: "un-reports-bucket"
  
corpus:
  target_documents: 10000      # Much larger scale
  full_text_search: true       # Enable PostgreSQL full-text search
```

3. **Update Code**
```python
# New files to create:
src/
├── db/
│   ├── __init__.py
│   ├── models.py           # SQLAlchemy models
│   ├── connection.py       # Database connection
│   └── migrations/         # Schema migrations
├── storage/
│   ├── __init__.py
│   ├── s3_handler.py      # File storage
│   └── local_handler.py   # Local fallback
└── search/
    ├── __init__.py
    ├── vector_search.py    # Vector similarity
    └── hybrid_search.py    # Combined search
```

#### Deployment Options

**Small Scale (1-10K documents):**
- **Database:** Managed PostgreSQL (Render, Railway, Neon)
- **App:** Streamlit/FastAPI on cloud platforms
- **Storage:** Database or small S3 bucket
- **Cost:** $20-50/month

**Medium Scale (10K-100K documents):**
- **Database:** AWS RDS PostgreSQL with pgvector
- **App:** ECS/Kubernetes deployment
- **Storage:** S3 with CloudFront CDN
- **Cost:** $100-500/month

**Large Scale (Full UN Corpus):**
- **Database:** Amazon Aurora PostgreSQL with replicas
- **Search:** Elasticsearch + vector search
- **App:** Kubernetes with auto-scaling
- **Cost:** $500-2000/month

### Option 3: Serverless Architecture

**Good for:** Variable usage patterns, cost optimization

```yaml
# Serverless stack
compute: AWS Lambda / Vercel Functions
database: PlanetScale / Neon Serverless
storage: S3 / Cloudflare R2
search: Pinecone / Weaviate Cloud
monitoring: Sentry / DataDog
```

## Migration Strategies

### Gradual Migration

1. **Phase 1:** Keep current local setup, add database as cache
2. **Phase 2:** Move embeddings to database, keep files local
3. **Phase 3:** Move to full database + object storage
4. **Phase 4:** Add advanced features (reranking, full-text search)

### Configuration-Based Scaling

```python
# Unified config approach
if config.get('database', {}).get('enabled'):
    from db.vector_store import DatabaseVectorStore
    vector_store = DatabaseVectorStore(config)
else:
    from index import FAISSIndexer
    vector_store = FAISSIndexer(config)
```

## Cost Estimations

### Demo Setup (Current)
- **Storage:** Free (GitHub)
- **Compute:** Free tier (Streamlit Community Cloud)
- **API:** $5-20/month (OpenAI, depending on usage)
- **Total:** $5-20/month

### Small Database Setup
- **Database:** $15/month (Neon/PlanetScale)
- **Compute:** $10/month (Railway/Render)
- **Storage:** $5/month (S3)
- **API:** $20-50/month (higher usage)
- **Total:** $50-80/month

### Production Setup
- **Database:** $100-300/month (managed PostgreSQL)
- **Compute:** $50-200/month (container hosting)
- **Storage:** $20-50/month (S3 + CDN)
- **API:** $100-500/month (high usage)
- **Total:** $270-1050/month

## Performance Considerations

### Local Files (Current)
- **Query time:** 100-500ms
- **Concurrent users:** 1-10
- **Data freshness:** Manual updates

### Database Backend
- **Query time:** 50-200ms (with proper indexes)
- **Concurrent users:** 100-1000+
- **Data freshness:** Real-time updates
- **Search quality:** Better (hybrid search)

## Getting Started with Database

### Quick Start with Docker

```bash
# Clone the repo
git clone your-repo
cd un-reports-rag

# Start database
docker-compose up -d postgres

# Enable database mode
echo "DATABASE_URL=postgresql://..." >> .env

# Run migration
python scripts/migrate_to_database.py

# Start app with database
streamlit run src/app.py
```

### Environment Variables

```bash
# .env file for database setup
DATABASE_URL=postgresql://user:pass@localhost:5432/unrag
OPENAI_API_KEY=sk-...
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
S3_BUCKET=your-bucket
REDIS_URL=redis://localhost:6379  # for caching
```

## Decision Matrix

| Factor | Local Files | Small DB | Production DB |
|--------|-------------|----------|---------------|
| Setup Time | 5 minutes | 30 minutes | 2-4 hours |
| Cost/Month | $5-20 | $50-80 | $270-1050 |
| Documents | 50-150 | 500-5000 | 10K-100K+ |
| Users | 1-10 | 10-100 | 100-1000+ |
| Updates | Manual | Automated | Real-time |
| Search Quality | Good | Better | Best |
| Maintenance | Minimal | Low | Medium-High |

## Conclusion

**Start with the demo setup** for most use cases. The local file approach with ~150 high-impact reports provides excellent value for research, education, and proof-of-concept purposes.

**Upgrade to database** when you need:
- More than 500 documents
- Multiple concurrent users
- Real-time updates
- Advanced search features

The modular design allows gradual migration without rewriting the entire system.