# complivibe-core

Regulatory compliance intelligence engine for the **EU AI Act** and **India DPDP Act 2023**.

---

## Project Structure

```
complivibe-core/
├── .github/
│   └── workflows/          # CI/CD (GitHub Actions)
├── backend/
│   ├── ingestion/          # Week 1 — regulatory text pipeline
│   ├── classifier/         # Week 2-3 — Annex III high-risk classifier
│   ├── mapper/             # Week 3-4 — EU AI Act + DPDP cross-mapper
│   ├── api/                # FastAPI routes
│   └── core/               # Config, DB connections, utils
├── frontend/               # Week 4+ — React dashboard
├── data/
│   ├── raw/                # Downloaded regulatory texts
│   ├── processed/          # Chunked + embedded text
│   └── mappings/           # Cross-regulation mapping tables (JSON/CSV)
├── scripts/                # One-off utility scripts
├── tests/
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production only — protected |
| `dev` | Active development — all PRs merge here |
| `feature/xxx` | Individual features |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for running Qdrant)

### Local development

```bash
# 1. Clone and enter the repo
git clone https://github.com/adarshkumar23/complivibe-core.git
cd complivibe-core

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment variables
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY etc.

# 5. Start the API server
uvicorn backend.api.main:app --reload
# API docs: http://localhost:8000/docs
```

### Docker Compose (full stack)

```bash
docker compose up --build
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Roadmap

| Week | Module | Goal |
|------|--------|------|
| 1 | `backend/ingestion` | Ingest & chunk EU AI Act and DPDP raw text |
| 2-3 | `backend/classifier` | Annex III high-risk AI system classifier |
| 3-4 | `backend/mapper` | EU AI Act ↔ DPDP cross-mapping |
| 4+ | `frontend` | React compliance dashboard |
