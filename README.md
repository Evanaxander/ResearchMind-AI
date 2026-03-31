# ResearchMind AI — Document Intelligence Platform

> Multi-agent RAG platform for analyzing any uploaded document set.
> Works out of the box for reports, company filings, policies, technical docs,
> and research PDFs. Finance-specific enrichment is available as an optional mode.
> Runs locally with role-aware answers, citations, and audit logging.

---

## Demo

### Upload any PDF/report → searchable evidence corpus
After upload, ResearchMind automatically:
- Detects broad document type from content and filename
- Preserves table structure in chunks when available
- Indexes content in FAISS for retrieval
- Adds metadata to the knowledge graph
- Returns an enrichment summary (domain-aware)

### Ask a question → cited, grounded answer
```
Q: "What are the top risks and recommended actions in this report set?"

A: The most frequently cited risks are vendor concentration, delayed
  compliance milestones, and infrastructure cost volatility. The reports
  recommend quarterly risk reviews and phased mitigation by business unit...
  [risk_register.pdf, chunk 21] [ops_review_q4.pdf, chunk 9]
```

### Role-based responses
Same question, different roles, different answers:
- **Analyst** → full answer with all source chunks and agent trace
- **Portfolio Manager** → answer with citations, raw text hidden
- **Compliance Officer** → answer with regulatory compliance notes added
- **Executive** → one-paragraph summary only

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User / Client                         │
│         (Swagger UI · React Dashboard · API)             │
└────────────────────────┬────────────────────────────────┘
                         │ JWT Auth
┌────────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                          │
│         REST endpoints · Role middleware · CORS          │
└──────────┬─────────────┬──────────────┬─────────────────┘
           │             │              │
┌──────────▼──┐  ┌───────▼──────┐  ┌───▼──────────────┐
│  LangGraph  │  │  RAG Pipeline │  │  Knowledge Graph  │
│  4 Agents   │  │  FAISS + Emb  │  │  Neo4j           │
│  ─────────  │  │  ──────────── │  │  ─────────────── │
│  Planner    │  │  Domain-Aware │  │  Document nodes  │
│  Researcher │  │  Parser       │  │  SAME_COMPANY    │
│  Analyst    │  │  Table-aware  │  │  SAME_PERIOD     │
│  Synthesizer│  │  Chunking     │  │  CONTRADICTS     │
└──────────┬──┘  └───────┬──────┘  └───┬──────────────┘
           │             │              │
┌──────────▼─────────────▼──────────────▼─────────────────┐
│                  Mistral-7B via Ollama                    │
│              Local inference · No API key                 │
└──────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│                    Infrastructure                         │
│         Docker · Redis · Audit Log · .env config         │
└─────────────────────────────────────────────────────────┘
```

---

## Evaluation Results

Benchmarked on "Attention Is All You Need" using LLM-as-judge scoring.

| System | Faithfulness | Relevancy | Context Recall |
|---|---|---|---|
| Base LLM (no RAG) | 0.00 | 1.00 | N/A |
| RAG only | 1.00 | 0.96 | 0.94 |
| **Multi-Agent RAG** | **1.00** | **1.00** | **1.00** |

Key findings:
- Base LLM scores 0.00 faithfulness — hallucinates without document context
- Adding RAG brings faithfulness to 1.00 — every claim grounded in source
- Multi-Agent achieves perfect scores — Planner's sub-task decomposition ensures complete retrieval

Run the evaluation:
```bash
python evaluation/run_comparison.py
```

---

## Features

### Domain Modes
- Default mode is `general` for mixed report/document workloads
- Optional `finance` mode adds ticker/period extraction and finance-focused signals
- Toggle behavior via environment variables without code changes

### Frontend User Flow
- Landing experience shows authentication only
- Users sign up, then log in
- Successful login routes to the main dashboard
- Dashboard includes upload, scope selection, querying, citations, and logout

### Document Intelligence
- Structure-aware PDF parsing using `pdfplumber` — tables preserved as markdown
- Automatic document type detection for broad report categories
- Domain-aware enrichment summaries (general or finance mode)
- Metadata-rich chunking for better retrieval and traceability

### Multi-Agent Research Pipeline
Four specialized LangGraph agents working in sequence:
1. **Planner** — classifies query type (metric/comparison/risk/general) and generates targeted sub-queries
2. **Researcher** — runs FAISS searches, applies metadata filters, deduplicates evidence
3. **Analyst** — performs structured analysis (metrics/comparison/risk/general)
4. **Synthesizer** — writes professional answers with source citations

### Knowledge Graph (Neo4j)
- Every document becomes a node with metadata
- Automatic relationship detection: `SAME_COMPANY`, `SAME_PERIOD`, `UPDATES`
- Contradiction detection between documents — stored as `CONTRADICTS` edges
- Graph traversal for finding related documents the user didn't mention
- Visual graph exploration via Neo4j Browser

### Role-Based Access Control
Four enterprise roles with differentiated access:

| Role | Answer Depth | Raw Text | Analysis | Audit Access |
|---|---|---|---|---|
| Analyst | Full | Yes | Yes | Yes |
| Portfolio Manager | Summary | No | Yes | No |
| Compliance Officer | Full + notes | Yes | Yes | Yes |
| Executive | Brief (1 para) | No | No | No |

### Audit Trail
- Every query logged: who, what, when, which documents, how long
- Stored as append-only JSONL — production-ready format
- Aggregate stats: queries by role, average latency, active users
- Compliance-ready: full traceability for regulatory requirements

---

## Quick Start

### Prerequisites
1. Install [Ollama](https://ollama.com/download)
2. Pull Mistral: `ollama pull mistral`
3. Install [Neo4j Desktop](https://neo4j.com/download/) and start a local database

### Run locally

```bash
# Clone
git clone https://github.com/Evanaxander/ResearchMind-AI.git
cd ResearchMind-AI

# Virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set NEO4J_PASSWORD to your Neo4j password

# Start
uvicorn app.main:app --reload --port 8010

# Frontend (new terminal)
cd frontend
npm install
npm start
```

Open:
- Frontend: `http://localhost:3000`
- API docs: `http://127.0.0.1:8010/docs`

### Run with Docker

```bash
cp .env.example .env
# Edit .env with your Neo4j password
docker-compose up --build
```

---

## API Reference

### Authentication
```bash
# Register
POST /api/v1/auth/register
{"username": "alice", "password": "pass123", "role": "analyst"}

# Login → get JWT token
POST /api/v1/auth/login
{"username": "alice", "password": "pass123"}

# Use token in all requests
Authorization: Bearer <token>
```

### Documents
```bash
POST   /api/v1/upload              # Upload PDF/TXT/DOCX
GET    /api/v1/documents           # List all documents
DELETE /api/v1/documents/{doc_id}  # Delete a document
```

### Query
```bash
POST /api/v1/query
{
  "question": "What are the key risks and recommendations across these reports?",
  "top_k": 5,
  "doc_type_filter": "report"
}
```

Finance-mode example:
```bash
POST /api/v1/query
{
  "question": "What was Apple's revenue trend over 3 years?",
  "top_k": 5,
  "ticker_filter": "AAPL",
  "period_filter": "FY2024"
}
```

### Knowledge Graph
```bash
GET /api/v1/graph/overview              # Full document graph
GET /api/v1/graph/company/{ticker}      # All docs for a company
GET /api/v1/graph/contradictions        # Detected contradictions
GET /api/v1/graph/related/{doc_id}      # Related documents
```

### Audit
```bash
GET /api/v1/audit/recent   # Recent query log (analyst/compliance only)
GET /api/v1/audit/stats    # Aggregate statistics
```

---

## Project Structure

```
financeiq/
├── app/
│   ├── main.py                        # FastAPI entry point
│   ├── core/config.py                 # Settings from .env
│   ├── models/schemas.py              # Pydantic models
│   ├── middleware/audit.py            # Audit logging
│   ├── routers/
│   │   ├── health.py                  # GET /health
│   │   ├── upload.py                  # Document upload
│   │   ├── query.py                   # Query + audit endpoints
│   │   ├── graph.py                   # Knowledge graph endpoints
│   │   └── auth.py                    # Authentication endpoints
│   └── services/
│       ├── financial_parser.py        # Structure-aware PDF parser
│       ├── metric_extractor.py        # Financial metric extraction
│       ├── financial_agent.py         # LangGraph 4-agent pipeline
│       ├── graph_service.py           # Neo4j operations
│       ├── contradiction_agent.py     # Cross-doc contradiction detection
│       ├── rag_service.py             # FAISS vector store
│       ├── document_service.py        # Full upload pipeline
│       ├── query_service.py           # Query orchestration
│       └── auth_service.py            # JWT + role management
├── evaluation/
│   ├── eval_dataset.json              # Test questions
│   ├── run_eval.py                    # Single system evaluation
│   └── run_comparison.py              # 3-system baseline comparison
├── tests/
│   └── test_phase1.py                 # pytest suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API framework | FastAPI | Async, auto-docs, Pydantic validation |
| Agent orchestration | LangGraph | Stateful multi-agent graphs |
| Vector store | FAISS | Fast similarity search at scale |
| PDF parsing | pdfplumber | Preserves table structure |
| Embeddings | sentence-transformers | Local, no API key |
| LLM | Mistral-7B via Ollama | Local inference, zero cost |
| Knowledge graph | Neo4j | Document relationships + traversal |
| Authentication | JWT (python-jose) | Stateless, enterprise-standard |
| Evaluation | LLM-as-judge | Ollama-compatible, no API key |
| Containerization | Docker Compose | Single-command deployment |

---

## Failure Cases & Limitations

Understanding where the system breaks is as important as knowing where it works:

- **Ambiguous queries** — Vague questions produce poor Planner sub-tasks, reducing retrieval quality. Specific questions perform significantly better.
- **Table-heavy documents** — Very dense financial tables (100+ rows) can exceed chunk context limits, causing partial extraction.
- **Ticker detection** — Ticker extraction uses frequency analysis which can misidentify common abbreviations. Filename-based detection is more reliable.
- **Memory constraints** — Mistral-7B requires ~4.5GB RAM. On memory-constrained systems, use `phi3` (2GB) as a drop-in replacement.
- **Cross-period comparison** — Comparing metrics across fiscal periods requires both documents to be uploaded and indexed. The system cannot infer historical data.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `NEO4J_URI` | Neo4j connection URL | `bolt://localhost:7687` |
| `NEO4J_PASSWORD` | Neo4j password | `neo4j` |
| `ANALYSIS_DOMAIN` | Domain mode: `general` or `finance` | `general` |
| `ENABLE_FINANCIAL_ENRICHMENT` | Enable finance-specific extraction | `false` |
| `UPLOAD_DIR` | Document storage path | `./uploads` |
| `FAISS_INDEX_DIR` | FAISS index path | `./faiss_index` |
| `MAX_UPLOAD_SIZE_MB` | Max upload size | `20` |
| `CHUNK_SIZE` | Characters per chunk | `512` |
| `CHUNK_OVERLAP` | Chunk overlap | `64` |

No LLM API keys required. All inference runs locally via Ollama.
