# backend/core

Orchestre les agents et compose la réponse finale.

- Ce qu’il fait: appelle Document, Similarité, Décision, Explication (et plus tard Image, Comportement, Fraude).
- Exemple simple:
  - Entrée: `CreditRequest`
  - Résultat attendu: `{"decision":"approve","explanation":"Decision=approve, fraud_ratio=0.2"}` (stub), puis format enrichi quand les agents seront branchés.
