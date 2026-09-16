import ollama

from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class Metadata:
    chunk_id: str
    page_no: str
    content_type: str
    section_title: str
    image_path: str
    source: str
    qdrant_score: float | None = None
    reranker_score: float | None = None


@dataclass
class Chunk:
    chunk_text: str
    metadata: Metadata
    embeddings: list[float] | None = None


class Embed:
    def __init__(self, model_id: str = "bge-m3"):
        self.model_id = model_id

    def _embed(self, text: str) -> list[float]:
        response = ollama.embed(
            model=self.model_id,
            input=text,
        )
        return response["embeddings"][0]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            chunk.embeddings = self._embed(chunk.chunk_text)

        return chunks


class Qdrant:
    def __init__(
        self,
        qdrant_output_dir: Path,
        collection_name: str,
    ):
        self.output_dir = qdrant_output_dir
        self.collection_name = collection_name
        self.client = self._create_qdrant_client()

    def _create_qdrant_client(self) -> QdrantClient:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        client = QdrantClient(path=str(self.output_dir))

        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                ),
            )

        return client

    def _recreate_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
        )

    def set(self, chunks: list[Chunk]) -> None:
        points: list[PointStruct] = []

        for chunk in chunks:
            if chunk.embeddings is None:
                raise RuntimeError(
                    f"[Qdrant] Embedding not generated for {chunk}"
                )

            points.append(
                PointStruct(
                    id=chunk.metadata.chunk_id,
                    vector=chunk.embeddings,
                    payload={
                        "source": chunk.metadata.source,
                        "page_no": chunk.metadata.page_no,
                        "content_type": chunk.metadata.content_type,
                        "section_title": chunk.metadata.section_title,
                        "chunk_text": chunk.chunk_text,
                        "image_path": chunk.metadata.image_path,
                    },
                )
            )

        self._recreate_collection()

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def get(
        self,
        query_embedding: list[float],
        limit: int = 20,
    ) -> list[Chunk]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        )

        retrieved_chunks: list[Chunk] = []

        for result in results.points:
            if result.payload is None:
                continue

            payload = result.payload

            retrieved_chunks.append(
                Chunk(
                    chunk_text=payload["chunk_text"],
                    metadata=Metadata(
                        chunk_id=str(result.id),
                        page_no=payload["page_no"],
                        content_type=payload["content_type"],
                        section_title=payload["section_title"],
                        image_path=payload["image_path"],
                        source=payload["source"],
                        qdrant_score=result.score,
                    ),
                )
            )

        return retrieved_chunks

    def close(self) -> None:
        self.client.close()


class Reranker:
    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.model = CrossEncoder(model_id)

    def rerank(
        self,
        query: str,
        documents: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        if not documents:
            return []

        pairs = [
            (query, document.chunk_text)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        for document, score in zip(documents, scores):
            document.metadata.reranker_score = float(score)

        documents.sort(
            key=lambda document: document.metadata.reranker_score or 0.0,
            reverse=True,
        )

        return documents[:top_k]
