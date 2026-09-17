import json

from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from rag.lib import Chunk, Embed, Metadata, Qdrant
from rag.config import config


def _load_chunks(mineru_output_dir: Path) -> list[Chunk]:
    print(f"[Ingestion] Searching for chunks in: {mineru_output_dir}")

    chunks: list[Chunk] = []
    files = list(mineru_output_dir.glob("*/output/kb_chunks.jsonl"))

    print(f"[Ingestion] Found {len(files)} JSONL files")

    for file in files:
        print(f"[Ingestion] Reading: {file}")

        file_chunks = 0

        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                chunk = Chunk(
                    chunk_text=data["chunk_text"],
                    metadata=Metadata(
                        chunk_id=str(
                            uuid5(
                                NAMESPACE_URL,
                                data["chunk_id"],
                            )
                        ),
                        source=data["chunk_id"],
                        page_no=data["page_no"],
                        content_type=data["content_type"],
                        section_title=data["section_title"],
                        image_path=data["image_path"],
                    ),
                )

                chunks.append(chunk)
                file_chunks += 1

        print(f"[Ingestion] Loaded {file_chunks} chunks from {file.name}")

    print(f"[Ingestion] Total chunks loaded: {len(chunks)}")
    return chunks


def ingestion_handler(
    mineru_output_dir: Path,
    qdrant_output_dir: Path,
    collection_name: str,
) -> None:
    """
    Ingest document chunks from MinerU output into a local Qdrant database.

    The function reads chunk data from JSONL files in the MinerU output
    directory, generates embeddings for each chunk, and stores the
    embedded chunks as vectors in the specified Qdrant collection.

    Args:
        mineru_output_dir: Directory containing MinerU document outputs
            and their `kb_chunks.jsonl` files.
        qdrant_output_dir: Directory used to persist the local Qdrant
            database.
        collection_name: Name of the Qdrant collection in which the
            embedded chunks are stored.

    Returns:
        None
    """

    print("[Ingestion] Starting ingestion pipeline")
    chunks = _load_chunks(mineru_output_dir)

    if not chunks:
        print("[Ingestion] No chunks found. Exiting.")
        return

    print(f"[Ingestion] Loaded {len(chunks)} chunks")
    embedder = Embed()
    qdrant = Qdrant()

    try:
        print("[Ingestion] Generating embeddings")
        embedded_chunks = embedder.embed_chunks(chunks)
        print(f"[Ingestion] Generated {len(embedded_chunks)} embeddings")
        print("[Ingestion] Storing vectors in Qdrant")
        qdrant.set(embedded_chunks)
        print(f"in Qdrant collection '{collection_name}'")

    finally:
        qdrant.close()
        print("[Ingestion] Ingestion complete")


if __name__ == "__main__":
    ingestion_handler(
        config.mineru_output_dir,
        config.qdrant_output_dir,
        config.qdrant_collection_name,
    )
