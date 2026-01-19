# backend/vector_db

Client Qdrant pour stocker/chercher des vecteurs.

- Ce qu’il fait: initialiser `QdrantClient` avec `QDRANT_URL` et permettre création/recherche de collections.
- Exemple simple:
  - Connexion: URL depuis `.env`.
  - Résultat attendu: opérations CRUD sur collections/vecteurs (à implémenter selon le schéma choisi).
