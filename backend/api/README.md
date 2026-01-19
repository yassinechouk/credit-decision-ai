# backend/api

Expose les routes HTTP et valide les données d’entrée.

- Ce qu’il fait: définit POST /credit/decision et le schéma `CreditRequest`.
- Exemple simple:
```
{
  "case_id": "CASE_001",
  "monthly_income": 2000,
  "requested_amount": 5000,
  "documents": ["invoice_001.pdf"]
}
```
- Résultat attendu: l’orchestrateur est appelé et renvoie la décision + explication.
