# Credit Decision AI

AI-assisted credit decision support with multi-agent analysis and human-in-the-loop review.

---

## What this repo contains

- **Backend (FastAPI):** orchestrates agents, stores credit cases, exposes REST APIs.
- **Frontend (React/Vite):** client and banker workflows (login, requests, decisions).
- **Vector search (Qdrant):** similarity lookup on historical credit cases.
- **PostgreSQL:** source of truth for cases, decisions, payments, and agent outputs.
- **Seed + synthetic data:** optional dataset for demos and similarity search.

---

## Quickstart (Docker Compose)

1) **Create env file**

```
cp .env.example .env
```

Fill in `OPENAI_API_KEY` if you want LLM-powered agents (optional). Defaults use an OpenAI-compatible base URL.

2) **Start services**

```
docker compose up --build
```

3) **Initialize database schema (first run only)**

Use the minimal base schema so the backend can create the payment tables on startup:

```
docker compose exec -T postgres psql -U postgres -d credit < data/sql/schema.sql
```

4) **Optional: apply migrations**

```
for f in migrations/*.sql; do
  docker compose exec -T postgres psql -U postgres -d credit < "$f"
done
```

5) **Optional: seed the database**

```
DB_HOST=localhost DB_PORT=5432 DB_NAME=credit DB_USER=postgres DB_PASSWORD=postgres \
  python seed_database.py
```

6) **Optional: load synthetic dataset into Qdrant**

```
docker compose exec backend python /app/data/synthetic/loadtoqdrant.py
```

**Ports**
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:4173`
- Qdrant: `http://localhost:6333`
- Postgres: `localhost:5432`

FastAPI docs are available at `http://localhost:8000/docs`.

---

## Local development (without Docker)

### Backend

```
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```
cd frontend
npm install
npm run dev
```

### Qdrant

Run locally (example):

```
docker run -p 6333:6333 qdrant/qdrant:v1.9.1
```

---

## Configuration

Main environment variables (see `.env.example`):

- `OPENAI_API_KEY`: enables LLM-backed agents (optional).
- `OPENAI_BASE_URL`: OpenAI-compatible endpoint (default: Groq).
- `LLM_MODEL`: default `llama-3.3-70b-versatile`.
- `QDRANT_URL`: default `http://localhost:6333`.
- `QDRANT_API_KEY`: optional.
- `QDRANT_COLLECTION_NAME`: default `credit_dataset`.
- `QDRANT_AUTO_LOAD`: set to `1` to auto-load dataset into Qdrant on startup.
- `SIMILARITY_DATASET_PATH`: path to `credit_dataset.json`.
- `EMBEDDING_MODEL`: default `sentence-transformers/all-MiniLM-L6-v2`.
- `TOP_K_SIMILAR`: number of similar cases returned (default 10).
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`.
- `UPLOAD_DIR`: where uploaded documents are stored (default `/app/data/uploads`).

---

## API overview

Authentication uses a simple bearer token returned by `/api/auth/login`.

Key endpoints:

- `POST /api/auth/login`
- `POST /api/client/credit-requests`
- `POST /api/client/credit-requests/upload` (multipart with files)
- `GET /api/client/credit-requests`
- `GET /api/client/credit-requests/{id}`
- `GET /api/banker/credit-requests`
- `GET /api/banker/credit-requests/{id}`
- `POST /api/banker/credit-requests/{id}/decision`
- `POST /api/banker/credit-requests/{id}/decision-suggestion`
- `POST /api/banker/credit-requests/{id}/rerun`
- `POST /api/banker/credit-requests/{id}/agent-chat`

### Example flow

```
# Login (seeded demo users)
# client1@test.com / hashed-password
# banker1@test.com / hashed-password
curl -X POST http://localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"client1@test.com","password":"hashed-password"}'

# Create a credit request (replace TOKEN)
curl -X POST http://localhost:8000/api/client/credit-requests \
  -H 'Authorization: Bearer TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "amount": 12000,
    "duration_months": 36,
    "monthly_income": 2500,
    "monthly_charges": 900,
    "employment_type": "employee",
    "contract_type": "permanent",
    "seniority_years": 3,
    "family_status": "single"
  }'
```

Sample documents are available in `data/sample_docs/`.

---

## Multi-agent design

Agents run independently and return structured outputs that are aggregated into a recommendation:

- **Document agent:** extracts signals from uploaded documents (LLM optional).
- **Similarity agent:** compares the case to historical profiles using Qdrant.
- **Behavior agent:** evaluates payment behavior signals (LLM optional).
- **Fraud agent:** flags anomalies (LLM optional).
- **Decision + Explanation agents:** consolidate signals into a recommendation and explanation.
- **Image agent:** stub heuristics for document quality (no real CV yet).

If `OPENAI_API_KEY` is not provided, agents fall back to heuristic or stub behavior.

---

## Frontends

- `frontend/`: primary React app for clients and bankers.
- `uiuxcredit2/`: UI/UX prototype bundle (standalone Vite app).

---

## Testing

Backend:

```
cd backend
pytest
```

Frontend:

```
cd frontend
npm run test:e2e
```

---

## Repository structure

```
├── backend/              FastAPI backend, agents, orchestration, tests
├── frontend/             React/Vite app (client + banker flows)
├── uiuxcredit2/          UI/UX prototype bundle
├── data/                 Synthetic datasets + sample docs
├── data/sql/schema.sql   Minimal base schema
├── migrations/           SQL migrations
├── docker-compose.yml    Local multi-service setup
├── schema.sql            Full schema (optional)
└── seed_database.py      Optional database seeding script
```

---

## Notes

- This project is a decision-support system, not an automated approval engine.
- Some agents are still heuristic or partial implementations.
- Use in production requires additional validation, compliance, and security hardening.
