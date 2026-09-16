import json

from pathlib import Path
from qdrant_client.models import PointStruct
from rag.lib import Chunk, Embed, Qdrant
from dataclasses import dataclass
from rag.settings import settings

def _load_chunks(mineru_output_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []

    for file in mineru_output_dir.glob("*/output/kb_chunks.jsonl"):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                chunks.append(Chunk(**json.loads(line)))

    return chunks


def ingestion_handler(mineru_output_dir: Path, qdrant_output_dir: Path, collection_name:str):
    """
    Embed document chunks and store the resulting vectors in local Qdrant.
    Args:
        mineru_output_dir:
            Root directory containing MinerU outputs.
        qdrant_output_dir:
            Directory where the local Qdrant database is stored.
        collection_name:
            qdrant collection name
    """
    print("[Ingestion] Loading chunks")
    embeder = Embed()
    qdrant = Qdrant(qdrant_output_dir, collection_name)
    chunks = _load_chunks(mineru_output_dir)

    if not chunks:
        print("[Ingestion] No chunks found.")
        return

    print(f"[Ingestion] Loaded {len(chunks)} chunks.")
    embedded_chunks = embeder.embed_chunks(chunks)
    print(f"[Ingestion] Generated {len(embedded_chunks)} embeddings.")
    qdrant.set(embedded_chunks)
    print(f"[Ingestion] Stored {len(embedded_chunks)} vectors in Qdrant collection '{collection_name}'.")
    qdrant.close()


if __name__ == "__main__":
    ingestion_handler(
        settings.mineru_output_dir,
        settings.qdrant_output_dir,
        settings.qdrant_collection_name
    )
