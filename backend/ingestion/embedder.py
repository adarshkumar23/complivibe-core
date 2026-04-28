from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Iterable, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from backend.core.config import config
from backend.ingestion.chunker import RegulationChunk


class RegulationEmbedder:
    def __init__(self) -> None:
        self._model = SentenceTransformer(config.embedding_model)
        self._client = chromadb.PersistentClient(
            path=config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.chroma_collection_regulations
        )

    def embed_chunks(
        self,
        chunks: Iterable[RegulationChunk],
        batch_size: int = 64,
    ) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            print("✓ embedder: no chunks to embed")
            return

        batches = [chunk_list[i : i + batch_size] for i in range(0, len(chunk_list), batch_size)]
        for batch in tqdm(batches, desc="Embedding", unit="batch"):
            batch_ids = [chunk.chunk_id for chunk in batch]

            try:
                existing = self._collection.get(ids=batch_ids, include=[])
                existing_ids = set(existing.get("ids", []))
            except Exception as exc:
                print(f"✗ embedder: failed to check existing ids: {exc}")
                existing_ids = set()

            to_embed = [chunk for chunk in batch if chunk.chunk_id not in existing_ids]
            if not to_embed:
                continue

            texts = [chunk.text for chunk in to_embed]
            try:
                embeddings = self._model.encode(texts, batch_size=len(texts), show_progress_bar=False)
                metadatas = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "regulation": chunk.regulation,
                        "article": chunk.article,
                        "chapter": chunk.chapter,
                        "page_start": chunk.page_start,
                    }
                    for chunk in to_embed
                ]

                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in to_embed],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
            except Exception as exc:
                print(f"✗ embedder: failed to upsert batch: {exc}")

        print("✓ embedder: finished")

    def search(
        self,
        query: str,
        top_k: int = 5,
        regulation: Optional[str] = None,
    ) -> list[dict]:
        if not query.strip():
            return []

        where_filter = {"regulation": regulation} if regulation else None

        try:
            query_embedding = self._model.encode([query], show_progress_bar=False)
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            print(f"✗ embedder: search failed: {exc}")
            return []

        return _format_query_results(results)

    def get_stats(self) -> dict[str, int]:
        try:
            payload = self._collection.get(include=["metadatas"])
            metadatas = payload.get("metadatas", [])
        except Exception as exc:
            print(f"✗ embedder: failed to fetch stats: {exc}")
            return {}

        counter: Counter[str] = Counter()
        for meta in metadatas:
            regulation = meta.get("regulation") if meta else None
            if regulation:
                counter[regulation] += 1

        return dict(counter)


def _format_query_results(payload: dict) -> list[dict]:
    ids = payload.get("ids", [[]])[0]
    documents = payload.get("documents", [[]])[0]
    metadatas = payload.get("metadatas", [[]])[0]
    distances = payload.get("distances", [[]])[0]

    results: list[dict] = []
    for item_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        results.append(
            {
                "id": item_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return results
