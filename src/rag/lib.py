import ollama

from pathlib import Path
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


@dataclass
class Chunk:
    chunk_id: str
    page_no: str
    content_type: str
    section_title: str
    chunk_text: str
    image_path: str


@dataclass
class EmbeddedChunk:
    chunk_id: str
    source: str
    page_no: str
    content_type: str
    section_title: str
    chunk_text: str
    image_path: str
    embedding: list[float]


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    source: str
    page_no: str
    content_type: str
    section_title: str
    chunk_text: str
    image_path: str


class Embed:
    def __init__(self, model_id: str = "bge-m3"):
        self.model_id = model_id

    def _embed(self, data: str) -> list[float]:
        response = ollama.embed(model=self.model_id, input=data)
        return response["embeddings"][0]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded_chunks: list[EmbeddedChunk] = []

        for chunk in chunks:
            result = self._embed(chunk.chunk_text)

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    source=chunk.chunk_id,
                    page_no=chunk.page_no,
                    content_type=chunk.content_type,
                    section_title=chunk.section_title,
                    chunk_text=chunk.chunk_text,
                    image_path=chunk.image_path,
                    embedding=result,
                )
            )

        return embedded_chunks

    
class Qdrant:
    def __init__(self, qdrant_output_dir: Path, collection_name: str):
        self.output_dir = qdrant_output_dir
        self.collection_name = collection_name

        self.client = self._create_qdrant_client()

    def _create_qdrant_client(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(self.output_dir))

        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

        return client

    def set(self, chunks: list[EmbeddedChunk]) -> None:
        points = []

        for chunk in chunks:
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload={
                        "source": chunk.source,
                        "page_no": chunk.page_no,
                        "content_type": chunk.content_type,
                        "section_title": chunk.section_title,
                        "chunk_text": chunk.chunk_text,
                        "image_path": chunk.image_path,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def get(self, query_embedding: list[float], limit: int = 20):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        )

        res = [
            {"score": point.score, **point.payload }
            for point in results.points
        ]
        print(res[0])
        return res


    def close (self) :
        self.client.close()
