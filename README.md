# Credit Decision AI

Minimal scaffold to experiment with a credit decision assistant. The stack uses FastAPI with stubbed agents and a Qdrant vector store running via Docker Compose.

## Quick start
- Prereqs: Docker + Docker Compose.
- Launch everything: `docker-compose up --build`
- FastAPI runs on http://localhost:8000 and Qdrant on http://localhost:6333.
- Frontend (static stub):
	- `cd frontend && npm install && npm run start`
	- Opens on http://localhost:4173 and calls the backend at http://localhost:8000.

## API
- POST /credit/decision
- Body example:
```
{
	"case_id": "CASE_001",
	"monthly_income": 2000,
	"requested_amount": 5000,
	"documents": ["invoice_001.pdf"]
}
```
- Response is stubbed: approve/review plus a short explanation.

## Project structure
```
credit-decision-ai/
├── docker-compose.yml
├── .env
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── orchestrator.py
│   │   └── config.py
│   ├── agents/
│   │   ├── document_agent.py
│   │   ├── similarity_agent.py
│   │   ├── decision_agent.py
│   │   └── explanation_agent.py
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   └── retrieval.py
│   └── vector_db/
│       └── qdrant_client.py
└── data/
		└── synthetic/
				└── sample_case.json
```

## Team rule
All new Python dependencies belong in backend/requirements.txt before use.