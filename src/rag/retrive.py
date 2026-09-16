from rag.lib import Embed, Qdrant, Reranker
from rag.settings import settings


def retrieve_answers(query: str, top_k: int = 5) -> list:
    """
    Retrieve and rerank relevant document chunks for a user query.

    The function generates an embedding for the query, retrieves the
    most similar chunks from Qdrant, reranks the retrieved chunks using
    a cross-encoder, and returns the top-ranked results.

    Args:
        query: User's question or search query.
        top_k: Number of top-ranked chunks to return after reranking.

    Returns:
        A list of the most relevant document chunks, including their
        Qdrant and reranker scores.
    """
    print(f"[Retrieve] Query: {query}")

    print("[Retrieve] Loading embedding model")
    embedder = Embed()

    print("[Retrieve] Connecting to Qdrant")
    qdrant = Qdrant(
        settings.qdrant_output_dir,
        settings.qdrant_collection_name,
    )

    print("[Retrieve] Loading reranker")
    reranker = Reranker()

    try:
        print("[Retrieve] Generating query embedding")
        query_embedding = embedder.embed_query(query)

        print("[Retrieve] Retrieving top 20 chunks from Qdrant")
        retrieved_chunks = qdrant.get(
            query_embedding,
            limit=20,
        )

        print(f"[Retrieve] Retrieved {len(retrieved_chunks)} chunks")

        print("[Retrieve] Reranking chunks")
        ranked_chunks = reranker.rerank(
            query=query,
            documents=retrieved_chunks,
            top_k=top_k,
        )

        print(f"[Retrieve] Selected top {len(ranked_chunks)} chunks")

        return ranked_chunks

    finally:
        print("[Retrieve] Closing Qdrant")
        qdrant.close()


if __name__ == "__main__":
    query = input("[Retrieve] Query: ").strip()

    if not query:
        raise SystemExit("[Retrieve] Query cannot be empty")

    results = retrieve_answers(query)

    print("\n[Retrieve] Final results:")

    for i, chunk in enumerate(results, 1):
        print(f"\n[Retrieve] --- Result {i} ---")
        print(f"[Retrieve] Qdrant score: {chunk.metadata.qdrant_score:.4f}")
        print(f"[Retrieve] Reranker score: {chunk.metadata.reranker_score:.4f}")
        print(f"[Retrieve] Source: {chunk.metadata.source}")
        print(f"[Retrieve] Page: {chunk.metadata.page_no}")
        print(f"[Retrieve] Section: {chunk.metadata.section_title}")
