from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

# ============================================
# 1. CONNEXION AU CLUSTER QDRANT
# ============================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "credit_dataset")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ============================================
# 2. CRÉER UNE COLLECTION
# ============================================

VECTOR_SIZE = 384  # Taille du modèle all-MiniLM-L6-v2

# Supprimer la collection si elle existe
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

# Créer la collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "profile": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        "payment": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    },
    on_disk_payload=True,
)

print(f"✅ Collection '{COLLECTION_NAME}' créée")

# Indexes payload (best-effort) to match the documented schema used by SimilarityAgentAI.
payload_schema = {
    "case_id": PayloadSchemaType.INTEGER,
    "loan_amount": PayloadSchemaType.FLOAT,
    "loan_duration": PayloadSchemaType.INTEGER,
    "monthly_income": PayloadSchemaType.FLOAT,
    "other_income": PayloadSchemaType.FLOAT,
    "monthly_charges": PayloadSchemaType.FLOAT,
    "employment_type": PayloadSchemaType.KEYWORD,
    "contract_type": PayloadSchemaType.KEYWORD,
    "seniority_years": PayloadSchemaType.INTEGER,
    "marital_status": PayloadSchemaType.KEYWORD,
    "number_of_children": PayloadSchemaType.INTEGER,
    "spouse_employed": PayloadSchemaType.BOOL,
    "housing_status": PayloadSchemaType.KEYWORD,
    "is_primary_holder": PayloadSchemaType.BOOL,
    "defaulted": PayloadSchemaType.BOOL,
    "fraud_flag": PayloadSchemaType.BOOL,
}

for field_name, field_schema in payload_schema.items():
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=field_schema,
        )
    except Exception:
        pass

# ============================================
# 3. CHARGER LE MODÈLE D'EMBEDDING
# ============================================

model = SentenceTransformer('all-MiniLM-L6-v2')

# ============================================
# 4. FONCTION POUR CONVERTIR UN RECORD EN TEXTE
# ============================================

def record_to_text(record:  dict) -> str:
    """
    Convertit un enregistrement de crédit en texte pour l'embedding. 
    """
    text = f"""
    Demande de prêt: 
    - Montant: {record['loan_amount']}€
    - Durée: {record['loan_duration']} mois
    - Revenu mensuel: {record['monthly_income']}€
    - Autres revenus: {record['other_income']}€
    - Charges mensuelles: {record['monthly_charges']}€
    - Type d'emploi: {record['employment_type']}
    - Type de contrat: {record['contract_type']}
    - Ancienneté: {record['seniority_years']} ans
    - Statut marital: {record['marital_status']}
    - Nombre d'enfants: {record['number_of_children']}
    - Conjoint employé: {record['spouse_employed']}
    - Statut logement: {record['housing_status']}
    - Titulaire principal: {record['is_primary_holder']}
    """
    return text.strip()

# ============================================
# 5. CHARGER ET VECTORISER LES DONNÉES
# ============================================

# Charger le fichier JSON
dataset_path = os.getenv(
    "SIMILARITY_DATASET_PATH",
    "/workspaces/credit-decision-ai/data/synthetic/credit_dataset.json",
)
with open(dataset_path, 'r') as f:
    credit_data = json.load(f)

# Préparer les points pour Qdrant
points = []

for record in credit_data: 
    # Créer le texte à vectoriser
    text = record_to_text(record)
    
    # Générer l'embedding
    vector = model.encode(text).tolist()
    
    # Créer le point Qdrant
    point = PointStruct(
        id=record['case_id'],
        vector={"profile": vector},
        payload={
            "case_id": record['case_id'],
            "loan_amount":  record['loan_amount'],
            "loan_duration": record['loan_duration'],
            "monthly_income": record['monthly_income'],
            "other_income": record['other_income'],
            "monthly_charges": record['monthly_charges'],
            "employment_type": record['employment_type'],
            "contract_type": record['contract_type'],
            "seniority_years": record['seniority_years'],
            "marital_status": record['marital_status'],
            "number_of_children": record['number_of_children'],
            "spouse_employed": record['spouse_employed'],
            "housing_status": record['housing_status'],
            "is_primary_holder":  record['is_primary_holder'],
            "defaulted": record['defaulted'],
            "fraud_flag": record['fraud_flag']
        }
    )
    points.append(point)

# ============================================
# 6. INSÉRER LES DONNÉES DANS QDRANT
# ============================================

# Insertion par batch (recommandé pour de gros volumes)
BATCH_SIZE = 100

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i: i + BATCH_SIZE]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )
    print(f"✅ Batch {i//BATCH_SIZE + 1} inséré ({len(batch)} points)")

print(f"\n🎉 Total:  {len(points)} enregistrements insérés dans Qdrant")
