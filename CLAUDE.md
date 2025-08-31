# Project Brief: UN Reports RAG (Past Year Only) — Minimal Demo

You’re my agentic coding partner. Please plan and scaffold a minimal end-to-end Python project that lets non-technical users chat over UN public reports from the past 12 months.  

**Priorities:** simplest stack, fewest moving parts, repeatable ingestion, clear citations, and easy one-click-ish deploy.

---

## Scope & Constraints

- **Corpus:** UN public “reports” published/issued in the last 365 days. Start with English; keep metadata so other languages can be added later.  
- **Data sources:** Prefer [United Nations Digital Library](https://digitallibrary.un.org) (UNDL) for record discovery + canonical links to files (often ODS). Avoid scraping research.un.org. Respect robots and throttle requests.  

### Minimal stack

- **Python:** 3.11  
- **Ingestion:** `requests`, `tqdm`, `pandas`  
- **Parsing:** `pymupdf` (PDFs), `trafilatura` (HTML)  
- **Embeddings:** OpenAI `text-embedding-3-small` (configurable), with local fallback `bge-small-en-v1.5`  
- **Vector store:** FAISS persisted to disk  
- **UI:** Streamlit chat (single-file app)  
- **Optional:** light reranker (e.g., `bge-reranker-base`)  
- **Deployment:** Streamlit Community Cloud or Hugging Face Space. No external DB.  
- **Citations:** Every answer must show *title, organ, date, UN symbol, and stable source URL* (UNDL or ODS).  
- **Politeness:** Use crawl delay (≥5s between requests), pagination, and cache record pages locally to avoid re-hitting servers.  

---

## Deliverables

un-rag/
README.md
requirements.txt
config.yaml
src/
discover.py # find last-year “report” records via UNDL
fetch.py # download files, store under data/raw
parse.py # pdf/html → text with headings; write JSONL/Parquet
index.py # embed chunks → FAISS; persist index + metadata
app.py # Streamlit chat with RAG + citations
eval.py # tiny sanity-check Q/A and offline smoke tests
utils.py # shared helpers (rate limit, caching, logging)
data/
raw/ # downloaded PDFs/HTML
parsed/ # chunks.parquet (or jsonl)
index.faiss
index.meta.json
scripts/
build_all.sh # one-liner: discover → fetch → parse → index
.env.example
Makefile
.gitignore


---

## Config (`config.yaml`)

- **Date window:** last 365 days, computed at runtime  
- **Language whitelist:** `["en"]` initially  
- **Record filters:** focus on items clearly marked as Reports  
- **Throttling:** `delay_seconds: 5`  
- **Retrieval:** `chunk_tokens: 1200`, `overlap: 150`, `top_k: 10`  
- **Embeddings provider:** `openai` (default) or `local_bge`  
- **Reranker:** none (default) or `bge-reranker-base`  

---

## Discovery Logic (`discover.py`)

- Identify the most robust, publicly accessible way to list UNDL records tagged as “Report” within the last year.  
- Prefer query params or JSON endpoints; if unavailable, parse record pages (not `/search` listings) in a compliant, throttled way.  
- Extract per record: `title`, `symbol`, `date`, `organ/body`, `language`, `record_url`, file URLs.  
- Deduplicate by UN symbol when present.  
- Persist as `records.parquet`.  

---

## Fetch (`fetch.py`)

- Download only needed language versions (start with English).  
- Save under `data/raw/{symbol or safe_id}.{lang}.pdf`.  
- Maintain a fetch manifest (`files.parquet`) with status, HTTP code, size, checksum, path.  

---

## Parse (`parse.py`)

- **PDFs:** use `pymupdf`; retain headings/section cues when possible.  
- **HTML:** use `trafilatura`.  
- Output chunked text with metadata: doc_id, symbol, title, date, organ, language, source_url, chunk_id, text


- Write to `data/parsed/chunks.parquet`.  

---

## Index (`index.py`)

- Embed chunks with selected provider.  
- Build FAISS index; persist as `index.faiss` and `index.meta.json` (embedding dims, provider, created_at).  
- Rebuild only when `chunks.parquet` changes.  

---

## RAG App (`app.py`)

Streamlit chat interface:

- **Sidebar:** corpus stats (docs, last update), filters (organ, date range), “Rebuild index” button.  
- **Main:** chat input; responses = concise answer + 2–4 citations (title, symbol, date, organ, link).  
- **Prompting:** forbid fabrications; require citations; fallback = guidance + links.  
- **Retrieval:** top-k semantic + optional rerank; dedupe by symbol; truncate context to model token budget.  

---

## Eval (`eval.py`)

- 10–20 test queries (e.g., “What did the Secretary-General report on [topic] in [month]?”).  
- Checks: non-empty answer, ≥1 citation, passage overlap with source.  

---

## Automation

- `scripts/build_all.sh` runs full pipeline.  
- `Makefile` targets: `make build`, `make app`, `make clean`.  
- GitHub Actions workflow (optional) to update monthly and re-commit index artifacts.  

---

## Secrets & Config

- Use `.env` for OpenAI keys; provide `.env.example`.  
- Fail gracefully if no API key and use local_bge fallback.  

---

## Acceptance Criteria

- `make build` works on pilot set (50–200 reports) within free-tier limits.  
- `streamlit run src/app.py` launches chat; answers cite UNDL/ODS links.  
- No hard dependency on external DBs/cloud beyond embedding API.  
- Respectful crawling (≥5s delay), resumable fetches, cached manifests.  

---

## Extras (Stretch Goals)

- Toggle to show PDF snippet around quoted text.  
- Simple corpus dashboard (docs by month, by organ, by series prefix).  

---

## REVISED IMPLEMENTATION PLAN

### Discovery Strategy (Post-Feasibility Analysis)
- **Primary**: Use UN body RSS feeds + sitemap crawling (respects robots.txt 5s delay)
- **Fallback**: Manual curation of high-value recent reports
- **Compliance**: `/search` endpoint blocked, so using allowed sitemap + individual record access
- **Target**: Security Council, General Assembly, ECOSOC recent reports

### Implementation Phases

**Phase 1: Core RAG Pipeline (MVP)**
1. Scaffold project structure with requirements.txt
2. Implement simplified discovery with curated seed data (20-50 reports)
3. Build fetch → parse → index → chat vertical slice
4. Deploy working Streamlit app

**Phase 2: Automated Discovery**
1. Add sitemap-based discovery for scale
2. Implement RSS feed monitoring
3. Add full automation with proper throttling

### Technical Decisions
- ✅ OpenAI API key configured as environment variable
- ✅ Use hybrid discovery (RSS + sitemap + manual seeds)
- ✅ 5-second delay compliance with robots.txt
- ✅ FAISS local storage (no external DB dependency)

## First Steps

1. ✅ Confirmed UNDL access patterns and robots.txt compliance
2. Create project scaffold with all modules
3. Implement vertical slice with seed data
4. Scale to full automation  


