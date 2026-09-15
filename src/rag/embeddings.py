import json
import logging
import ollama

from typing import Any
from pydantic import BaseModel, Field
from rag.config import get_output_dir


logger = logging.getLogger(__name__)

d = get_output_dir()
output_file = d / "embeddings.json"
input_file = d / "chunks.json"

EMBEDDING_MODEL = "bge-m3"

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

def load_chunks() -> list[Chunk]:
    with open(input_file, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    return [Chunk.model_validate(chunk) for chunk in data]


def create_embedding() -> None:
    chunks = load_chunks()
    embedded_chunks: list[dict[str, object]] = []

    logger.info(f"Creating embeddings for {len(chunks)} chunks")

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Embedding chunk {i}/{len(chunks)}")
        response = ollama.embed(model=EMBEDDING_MODEL, input=chunk.page_content)
        embedding: list[float] = response["embeddings"][0]
        embedded_chunks.append({
            "page_content": chunk.page_content,
            "metadata": chunk.metadata.model_dump(),
            "embeddings": embedding,
        })
        logger.debug(f"Chunk {i}: {len(chunk.page_content)} chars → {len(embedding)} dimensions")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote embeddings to {output_file}")

if __name__ == "__main__":
    create_embedding()
