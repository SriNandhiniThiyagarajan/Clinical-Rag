from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLLECTION_NAME = "clinical_chunks"
VECTOR_SIZE = 384

client = QdrantClient(host="localhost",port =6333)

#delete if exists
existing = [c.name for c in client.get_collections().collections]

if COLLECTION_NAME in existing:
    client.delete_collection(
        collection_name =COLLECTION_NAME
    )
    print(f"{COLLECTION_NAME} deleted successfully")

client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config = qm.VectorParams(size=VECTOR_SIZE,distance=qm.Distance.COSINE)

)
print(f"{COLLECTION_NAME} recreated sucessfully")