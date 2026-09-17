import ollama

from google import genai
from pathlib import Path
from google.genai import types
from rag.config import config
from dataclasses import dataclass
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
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
    def __init__(self):
        self.model = SentenceTransformer(config.embedding_model_id)

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=config.embedding_normalize
        )
        return embedding.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        texts = [chunk.chunk_text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=config.embedding_batch_size,
            normalize_embeddings=config.embedding_normalize,
            show_progress_bar=config.embedding_show_progress,
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embeddings = embedding.tolist()

        return chunks


class Qdrant:
    def __init__(self):
        self.output_dir = config.qdrant_output_dir
        self.collection_name = config.qdrant_collection_name
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
    def __init__(self):
        self.model = CrossEncoder(config.reranker_model_id)
        self.top_k = config.reranker_top_k

    def rerank(
        self,
        query: str,
        documents: list[Chunk],
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

        return documents[:self.top_k]


class LLM:
    def __init__(self):
        self.model_id = config.llm_model_id
        self.client = genai.Client(api_key=config.google_api_key)

    def to_text(self, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant context found."
            
        context = []
        for i, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata
            context.append(
                "\n".join([
                    f"[Context {i}]",
                    f"Source: {metadata.source}",
                    f"Page: {metadata.page_no}",
                    f"Section: {metadata.section_title}",
                    "Content:",
                    chunk.chunk_text,
                ])
            )
        return "\n\n".join(context)

    def generate_answer(self, query: str, chunks: list[Chunk]) -> str:
        context = self.to_text(chunks)

        system_prompt = (
            "You are a question-answering assistant for a RAG system.\n\n"
            "Answer the user's question using only the provided context.\n\n"
            "Rules:\n"
            "- Do not invent information.\n"
            '- If the context does not contain enough information, say: "I don\'t know based on the provided documents."\n'
            "- Give a clear and concise answer.\n"
            "- Do not generate or modify source information."
        )

        contents = f"Context:\n{context}\n\nUser Question:\n{query}"

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                ),
            )
            return response.text if response.text else "No response generated by the model."
            
        except Exception as e:
            print(f"An error occurred while generating the answer: {str(e)}")
            raise e
