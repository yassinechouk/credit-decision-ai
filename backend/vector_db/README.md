# backend/vector_db

Client Qdrant pour stocker/chercher des vecteurs.

- Ce qu’il fait: initialiser `QdrantClient` avec `QDRANT_URL` et permettre création/recherche de collections.
- Exemple simple:
  - Connexion: URL depuis `.env`.
  - Résultat attendu: opérations CRUD sur collections/vecteurs (à implémenter selon le schéma choisi).

## Collection utilisee par SimilarityAgentAI

- Collection: `credit_dataset`
- Distance: cosine
- Vector size: dimension du modele d'embedding (ex: 384 pour `all-MiniLM-L6-v2`)
- Vectors: multi-vector (mini) `profile` + `payment` (meme dimension)
- Payload: "on-disk" active (pour economiser la RAM) + indexes payload (best-effort)

### Schema payload (compatible dataset actuel)

Ces champs existent dans `data/synthetic/credit_dataset.json` et sont utilises comme payload Qdrant.
Ils sont aussi documentes dans `backend/agents/similarity_agent.py`.

- `case_id` (integer)
- `loan_amount` (float)
- `loan_duration` (integer)
- `monthly_income` (float)
- `other_income` (float)
- `monthly_charges` (float)
- `employment_type` (keyword)
- `contract_type` (keyword)
- `seniority_years` (integer)
- `marital_status` (keyword)
- `number_of_children` (integer)
- `spouse_employed` (bool)
- `housing_status` (keyword)
- `is_primary_holder` (bool)
- `defaulted` (bool)
- `fraud_flag` (bool)

### Variables d'environnement

- `QDRANT_URL`
- `QDRANT_API_KEY` (optionnel)
- `QDRANT_AUTO_LOAD=1` pour charger automatiquement le dataset dans Qdrant au demarrage
- `SIMILARITY_DATASET_PATH` (optionnel) pour pointer vers un dataset different

### Recherche par type de vecteur

- `vector_type="profile"` (par defaut): similarite globale du profil
- `vector_type="payment"`: similarite basee sur comportement de paiement (si disponible)
