# Credit Decision AI

Minimal scaffold to experiment with a credit decision assistant. The stack uses FastAPI with stubbed agents, a Qdrant vector store, and a small static frontend.

## Rôle général du projet (en simple)
Tu es un agent qui aide les analystes crédit. Tu compares toujours le dossier actuel à la mémoire (cas historiques) et tu expliques tes raisons. La fraude est un signal parmi d’autres, jamais l’unique objectif. L’humain garde la décision finale.

## Quick start
- Prérequis : Docker + Docker Compose.
- Lancer tout : `docker-compose up --build`
- Backend FastAPI : http://localhost:8000
- Qdrant : http://localhost:6333
- Frontend statique :
  - `cd frontend && npm install && npm run start`
  - Ouvre sur http://localhost:4173 et appelle le backend en 8000.

## Ce que fait chaque dossier (avec un exemple attendu)
- docker-compose.yml : lance backend + Qdrant. Exemple attendu : `docker-compose up --build` expose l’API en 8000 et Qdrant en 6333.
- .env : variables partagées (URL Qdrant, modèle d’embedding). Exemple attendu : le backend lit `QDRANT_URL` pour se connecter.
- backend/ : API FastAPI + orchestrateur + agents.
  - api/ : routes et schémas d’entrée. Exemple : POST /credit/decision accepte un corps JSON de demande de crédit.
  - core/ : orchestrateur qui appelle les agents et assemble la réponse.
  - agents/ : logique par rôle (stubs pour démarrer).
  - rag/ : briques de chunking/embedding/retrieval (à connecter à Qdrant).
  - vector_db/ : client Qdrant.
- frontend/ : page statique qui envoie une demande à l’API (pour tester rapidement).
- data/synthetic/ : exemples de cas (jeu de test minimal).

## Rôles des agents (simples et séparés)
- Agent Document : décrit les infos des documents (montants, revenus, cohérence), n’accuse pas.
  - Exemple de sortie : {"document_quality":"good","income_stability":"medium","extracted_monthly_income":2800,"flags":["revenu variable"]}
- Agent Image : juge la crédibilité visuelle (scan, signatures, mise en page), pas la morale.
  - Exemple : {"visual_quality":"acceptable","document_consistency":"high","flags":["scan basse résolution"]}
- Agent Comportement : évalue le sérieux (remplissage, corrections, régularité).
  - Exemple : {"behavior_profile":"normal","confidence_level":"high"}
- Agent Similarité : compare au passé (bons et mauvais rembourseurs).
  - Exemple : {"similar_good_profiles":14,"similar_bad_profiles":3,"repayment_success_rate":0.82,"insight":"profil proche de freelances stables"}
- Agent Fraude : signale des indices de fraude documentaire.
  - Exemple : {"fraud_risk":"low","fraud_probability":0.12}
- Agent Décision : agrège tout et recommande (approve/review/decline) avec raisons.
  - Exemple : {"credit_eligibility":"approved","recommended_amount":15000,"confidence":"high","main_reasons":["revenus stables","documents crédibles","profil similaire à clients solvables"]}

## Exemple de requête et de réponse attendue
Requête POST /credit/decision :
```
{
  "case_id": "CASE_001",
  "monthly_income": 2000,
  "requested_amount": 5000,
  "documents": ["invoice_001.pdf"]
}
```
Réponse stub actuelle (pour démarrer) :
```
{
  "decision": "approve",
  "explanation": "Decision=approve, fraud_ratio=0.2"
}
```
Réponse cible (quand les agents seront branchés) :
```
{
  "credit_eligibility_recommendation": "approve",
  "recommended_amount": 15000,
  "confidence_level": "high",
  "main_reasons": ["revenus stables", "documents crédibles", "profil similaire à clients solvables"],
  "uncertainty_factors": ["revenus légèrement variables"],
  "historical_reference_summary": "proche de 14 freelances ayant bien remboursé; peu de cas similaires en défaut"
}
```

## Project structure
````
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
````

## Team rule
All new Python dependencies belong in backend/requirements.txt before use.
