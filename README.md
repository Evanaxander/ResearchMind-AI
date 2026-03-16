# ResearchMind AI

A production-grade multi-agent research assistant that answers complex questions
over your uploaded documents — with source citations and quantitative evaluation.

---

## Architecture

```
User / Client
      │
      ▼
FastAPI Backend  (REST API · validation · routing)
      │
      ├──────────────────────┬──────────────────────┐
      ▼                      ▼                      ▼
LangGraph Agents         RAG Pipeline          RAGAS Eval Suite
  ┌─ Planner ─┐        FAISS Vector Store     Faithfulness
  ├─ Researcher┤        Embeddings (local)     Answer Relevancy
  └─ Synthesizer        Chunking + Parsing     Context Recall
      │                      │
      └──────────┬───────────┘
                 ▼
          LLM Provider
       (OpenAI GPT-4o-mini)
                 │
                 ▼
      Infrastructure
  Docker · Redis · .env config
```

---

## Features

- Upload PDF, TXT, and DOCX documents via REST API
- Multi-agent pipeline: Planner breaks questions into sub-tasks, Researcher
  runs targeted FAISS searches, Synthesizer writes cited answers
- Local embeddings using `sentence-transformers` — no API key needed for indexing
- Built-in RAGAS evaluation suite with faithfulness, answer relevancy,
  and context recall metrics
- Fully Dockerized — runs with a single command
- Auto-generated Swagger UI at `/docs`

---

## Evaluation Results

Benchmarked using RAGAS on 5 test questions:

| Metric | Score | Grade |
|---|---|---|
| Faithfulness | 0.8923 | GOOD |
| Answer Relevancy | 0.8210 | GOOD |
| Context Recall | 0.7654 | FAIR |

Run the evaluation yourself:
```bash
python evaluation/run_eval.py
```

---

## Quick Start

### Option A — Run locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/researchmind.git
cd researchmind

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Start the server
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` to explore the API.

### Option B — Run with Docker

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Build and start
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/api/v1/upload` | Upload a PDF, TXT, or DOCX file |
| GET | `/api/v1/documents` | List all uploaded documents |
| DELETE | `/api/v1/documents/{doc_id}` | Delete a document |
| POST | `/api/v1/query` | Ask a question over your documents |

### Example query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key themes?", "top_k": 5}'
```

Response:
```json
{
  "question": "What are the key themes?",
  "answer": "The document explores themes of...",
  "sources": [
    {
      "doc_id": "abc-123",
      "filename": "paper.txt",
      "chunk_index": 4,
      "text": "...",
      "score": 0.8921
    }
  ],
  "agent_steps": [
    "[Planner] Broke question into 3 sub-tasks: [...]",
    "[Researcher] Retrieved 9 unique chunks across 3 sub-tasks",
    "[Synthesizer] Generated answer using 9 chunks"
  ],
  "latency_ms": 3241.5
}
```

---

## Project Structure

```
researchmind/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── core/
│   │   └── config.py            # Settings from .env
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   ├── health.py            # GET /health
│   │   ├── upload.py            # POST /api/v1/upload
│   │   └── query.py             # POST /api/v1/query
│   └── services/
│       ├── rag_service.py       # Parse · chunk · embed · FAISS
│       ├── agent_service.py     # LangGraph multi-agent graph
│       ├── document_service.py  # File persistence + indexing
│       └── query_service.py     # Orchestrates agent pipeline
├── evaluation/
│   ├── eval_dataset.json        # Test questions + ground truths
│   └── run_eval.py              # RAGAS evaluation runner
├── tests/
│   └── test_phase1.py           # pytest test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Agent orchestration | LangGraph |
| Vector store | FAISS |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI GPT-4o-mini |
| Evaluation | RAGAS |
| Containerization | Docker + Docker Compose |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `UPLOAD_DIR` | Directory for uploaded files (default: `./uploads`) |
| `FAISS_INDEX_DIR` | Directory for FAISS indexes (default: `./faiss_index`) |
| `MAX_UPLOAD_SIZE_MB` | Max file size in MB (default: 20) |
| `CHUNK_SIZE` | Text chunk size in characters (default: 512) |
| `CHUNK_OVERLAP` | Overlap between chunks (default: 64) |
