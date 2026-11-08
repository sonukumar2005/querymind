"""
rag.py — ChromaDB-based RAG pipeline for QueryMind AI.

Uses the new `google.genai` SDK for embeddings.
"""

import os
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
VECTOR_STORE_DIR = Path("vector_store")
COLLECTION_NAME  = "schema_embeddings"

_genai_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _genai_client


# ── Embedding function wrapper (ChromaDB-compatible) ───────────────────────────

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """ChromaDB embedding function backed by Gemini text-embedding API."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        client = _get_client()
        embeddings = []
        for text in input:
            resp = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            embeddings.append(resp.embeddings[0].values)
        return embeddings


# ── SchemaRAG class ────────────────────────────────────────────────────────────

class SchemaRAG:
    """
    Lightweight RAG pipeline that embeds table schemas and retrieves
    the most relevant schema context for a user query.
    """

    def __init__(self):
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._embed_fn = GeminiEmbeddingFunction() if GOOGLE_API_KEY else None
        self._client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            if self._embed_fn:
                return self._client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self._embed_fn,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                return self._client.get_or_create_collection(name=COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to initialise ChromaDB collection: {e}")

    def _doc_id(self, table_name: str) -> str:
        return hashlib.md5(table_name.encode()).hexdigest()

    def _schema_to_document(self, table_name: str, schema_dict: dict) -> str:
        """Convert schema dict into natural-language description for embedding."""
        lines = [f"Table name: {table_name}", "Columns:"]
        for col in schema_dict.get("columns", []):
            pk = " (primary key)" if col.get("primary_key") else ""
            nn = " not null" if not col.get("nullable", True) else ""
            lines.append(f"  - {col['name']} [{col['type']}{pk}{nn}]")
        fks = schema_dict.get("foreign_keys", [])
        if fks:
            lines.append("Relationships:")
            for fk in fks:
                lines.append(
                    f"  - {', '.join(fk['constrained_columns'])} references "
                    f"{fk['referred_table']}({', '.join(fk['referred_columns'])})"
                )
        return "\n".join(lines)

    def index_schema(self, full_schema: dict[str, dict]):
        """Index (or re-index) all table schemas into ChromaDB."""
        if not full_schema or not self._embed_fn:
            return
        ids, documents, metadatas = [], [], []
        for table_name, schema_dict in full_schema.items():
            doc = self._schema_to_document(table_name, schema_dict)
            ids.append(self._doc_id(table_name))
            documents.append(doc)
            metadatas.append({"table": table_name})
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def get_relevant_schema(self, user_question: str, top_k: int = 3) -> str:
        """Retrieve most relevant table schemas for the given question."""
        if not self._embed_fn:
            return ""
        try:
            count = self._collection.count()
            if count == 0:
                return ""
            results = self._collection.query(
                query_texts=[user_question],
                n_results=min(top_k, count),
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [[]])[0]
            return "\n\n".join(docs) if docs else ""
        except Exception:
            return ""

    def refresh(self, full_schema: dict[str, dict]):
        """Re-index all schemas (called after any upload or schema change)."""
        self.index_schema(full_schema)

    def clear(self):
        """Delete all indexed schema documents."""
        try:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._get_or_create_collection()
        except Exception:
            pass


# ── Singleton instance ─────────────────────────────────────────────────────────

_rag_instance: SchemaRAG | None = None


def get_rag() -> SchemaRAG:
    """Return the singleton SchemaRAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = SchemaRAG()
    return _rag_instance
