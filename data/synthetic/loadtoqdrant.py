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

# ============================================
# 1. CONNEXION AU CLUSTER QDRANT
# ============================================

# Option A:  Qdrant Cloud (recommandé pour production)
client = QdrantClient(
    url="https://44775a69-b58f-449f-b5ca-b0f6ec6b5862.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.51Eobf7Ye3tWtM_4YRPqCtAAvPXIssDAJbgm3KHx9ic",
)

# ============================================
# 2. CRÉER UNE COLLECTION
# ============================================

COLLECTION_NAME = "credit_dataset"
VECTOR_SIZE = 384  # Taille du modèle all-MiniLM-L6-v2

# Supprimer la collection si elle existe
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

# Créer la collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE  # ou EUCLID, DOT
    )
)

print(f"✅ Collection '{COLLECTION_NAME}' créée")

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
with open('credit_dataset.json', 'r') as f:
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
        vector=vector,
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