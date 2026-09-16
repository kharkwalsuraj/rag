from rag.lib import Embed, Qdrant, Reranker, LLM
from rag.settings import settings


def retrieve_answers(query: str, top_k: int = 5):
    """
    Retrieve, rerank, and generate an answer for a user query.
    """
    print(f"[Retrieve] Query: {query}")

    embedder = Embed()
    qdrant = Qdrant(
        settings.qdrant_output_dir,
        settings.qdrant_collection_name,
    )
    reranker = Reranker()
    llm = LLM()

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

        print("[Retrieve] Generating LLM response")
        response = llm.generate_answer(
            query=query,
            chunks=ranked_chunks,
        )

        return { "answer" :response, "source": ranked_chunks}

    finally:
        qdrant.close()


if __name__ == "__main__":
    query = input("[Retrieve] Query: ").strip()

    if not query:
        raise SystemExit("[Retrieve] Query cannot be empty")

    response = retrieve_answers(query)

    print("\n[Retrieve] Final response:")
    print(response)
