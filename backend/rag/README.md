# backend/rag

Prépare la mémoire/similarité (chunking, embeddings, retrieval).

- Ce qu’il fait: découpe textes, calcule embeddings (SentenceTransformer), recherche des cas proches (Qdrant).
- Exemple simple:
  - Entrée: "Facture de 5000 EUR, paiement mensuel" -> `embed()` renvoie un vecteur.
  - Résultat attendu: `retrieve_similar()` renvoie des cas proches (stub pour l’instant, à brancher à Qdrant).
