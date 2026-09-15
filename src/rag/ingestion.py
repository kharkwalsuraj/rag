import json
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION_NAME = "knowledge_base"

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


def _load_chunks(mineru_output_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []

    for file in mineru_output_dir.glob("*/output/kb_chunks.jsonl"):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                chunks.append(Chunk(**json.loads(line)))

    return chunks


def _embed(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    embedded_chunks: list[EmbeddedChunk] = []

    for chunk in chunks:
        response = ollama.embed(model="bge-m3", input=chunk.chunk_text)
        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                source=chunk.chunk_id,
                page_no=chunk.page_no,
                content_type=chunk.content_type,
                section_title=chunk.section_title,
                chunk_text=chunk.chunk_text,
                image_path=chunk.image_path,
                embedding=response["embeddings"][0],
            )
        )

    return embedded_chunks


def _create_qdrant_client(qdrant_output_dir: Path) -> QdrantClient:
    qdrant_output_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(qdrant_output_dir))

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE,
        ),
    )

    return client


def embeddings_handler(
    mineru_output_dir: Path,
    qdrant_output_dir: Path,
) -> None:
    """
    Embed document chunks and store the resulting vectors in local Qdrant.

    Args:
        mineru_output_dir:
            Root directory containing MinerU outputs.

        qdrant_output_dir:
            Directory where the local Qdrant database is stored.
    """

    chunks = _load_chunks(mineru_output_dir)

    if not chunks:
        print("No chunks found.")
        return

    print(f"Loaded {len(chunks)} chunks.")
    embedded_chunks = _embed(chunks)
    print(f"Generated {len(embedded_chunks)} embeddings.")
    client = _create_qdrant_client(qdrant_output_dir)

    try:
        points = [
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
            for chunk in embedded_chunks
        ]

        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)

    finally:
        client.close()

    print(f"Stored {len(embedded_chunks)} vectors in Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    cwd = Path.cwd()

    mineru_output_dir = cwd / "output" / "mineru"
    qdrant_output_dir = cwd / "output" / "qdrant"

    embeddings_handler(mineru_output_dir, qdrant_output_dir,)
