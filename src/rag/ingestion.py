import json

from pathlib import Path
from qdrant_client.models import PointStruct
from rag.lib import Chunk, COLLECTION_NAME, embedd, create_qdrant_client


def _load_chunks(mineru_output_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []

    for file in mineru_output_dir.glob("*/output/kb_chunks.jsonl"):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                chunks.append(Chunk(**json.loads(line)))

    return chunks


def embeddings_handler(mineru_output_dir: Path, qdrant_output_dir: Path):
    """
    Embed document chunks and store the resulting vectors in local Qdrant.

    Args:
        mineru_output_dir:
            Root directory containing MinerU outputs.

        qdrant_output_dir:
            Directory where the local Qdrant database is stored.
    """
    print("[Ingestion] Loading chunks")
    chunks = _load_chunks(mineru_output_dir)

    if not chunks:
        print("[Ingestion] No chunks found.")
        return

    print(f"[Ingestion] Loaded {len(chunks)} chunks.")
    embedded_chunks = [embedd(chunk) for chunk in chunks]
    print(f"[Ingestion] Generated {len(embedded_chunks)} embeddings.")
    client = create_qdrant_client(qdrant_output_dir)

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

    print(f"[Ingestion] Stored {len(embedded_chunks)} vectors in Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    cwd = Path.cwd()

    mineru_output_dir = cwd / "output" / "mineru"
    qdrant_output_dir = cwd / "output" / "qdrant"

    embeddings_handler(mineru_output_dir, qdrant_output_dir,)




# TODO: fix whole code