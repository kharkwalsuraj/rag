import json
import logging
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rag.config import COLLECTION_NAME, get_output_dir


logger = logging.getLogger(__name__)


class Metadata(BaseModel):
    header1: str | None = None
    header2: str | None = None
    header3: str | None = None
    header4: str | None = None
    images: list[str] = Field(default_factory=list)
    images_description: list[str] = Field(default_factory=list)
    tables_html: list[str] = Field(default_factory=list)
    source: str


class Chunk(BaseModel):
    page_content: str
    metadata: Metadata
    embeddings: list[float]


client = QdrantClient(path="./output/vectordb")
INPUT_FILE = get_output_dir() / "embeddings.json"


def load_embeddings() -> list[Chunk]:
    logger.info("Loading embeddings from %s", INPUT_FILE)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    chunks = [Chunk.model_validate(chunk) for chunk in data]
    logger.info("Loaded %d chunks", len(chunks))

    return chunks


def create_collection(vector_size: int) -> None:
    if client.collection_exists(COLLECTION_NAME):
        logger.info("Deleting existing collection: %s", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    logger.info(
        "Creating collection '%s' with vector size %d",
        COLLECTION_NAME,
        vector_size,
    )

    _ = client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def store_embeddings(chunks: list[Chunk]) -> None:
    if not chunks:
        raise ValueError("No chunks found")

    vector_size = len(chunks[0].embeddings)

    if not all(len(chunk.embeddings) == vector_size for chunk in chunks):
        raise ValueError("Embedding dimensions are inconsistent")

    logger.info("Embedding dimension: %d", vector_size)

    create_collection(vector_size)

    points = [
        PointStruct(
            id=index,
            vector=chunk.embeddings,
            payload={
                "page_content": chunk.page_content,
                "metadata": chunk.metadata.model_dump(),
            },
        )
        for index, chunk in enumerate(chunks)
    ]

    logger.info(
        "Storing %d embeddings in collection '%s'",
        len(points),
        COLLECTION_NAME,
    )

    _ = client.upsert(collection_name=COLLECTION_NAME, points=points)

    logger.info("Successfully stored embeddings")


def main() -> None:
    chunks = load_embeddings()
    store_embeddings(chunks)

    collection = client.get_collection(COLLECTION_NAME)
    logger.info(f"Qdrant collection {COLLECTION_NAME} contains {collection.points_count} points")
    client.close()


if __name__ == "__main__":
    main()
