import os
from qdrant_client import QdrantClient

client = QdrantClient(url=os.getenv("QDRANT_URL"))
