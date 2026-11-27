import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


# V3 — Seleção de índice documental reconstruído

import os as _os
import chromadb as _chromadb

_v3_index_path = _os.getenv("DOR_V3_INDEX_PATH")

if _v3_index_path:

    _v3_client = _chromadb.PersistentClient(
        path=_v3_index_path
    )

    collection = _v3_client.get_collection(
        name="knowledge_base"
    )
