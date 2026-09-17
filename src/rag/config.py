from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """Configuration for the RAG application."""

    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "outputs")

    # MinerU
    mineru_token: str = Field(default="", validation_alias="MINERU_TOKEN")
    mineru_output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "outputs" / "mineru"
    )
    documents_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "tests" / "documents"
    )

    # Qdrant
    qdrant_collection_name: str = "knowledge_base"
    qdrant_output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "outputs" / "qdrant"
    )

    # Embedding
    embedding_model_id: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 32
    embedding_normalize: bool = True
    embedding_show_progress: bool = True

    # Reranker
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5

    # LLM
    google_api_key: str = Field(default="", validation_alias="GOOGLE_API")
    llm_model_id: str = "gemini-3.6-flash"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def health_check(self) -> None:
        """Check required configuration and runtime dependencies."""

        print("RAG Configuration")
        print("=" * 40)

        self._check_directories()
        self._check_environment()
        self._check_dependencies()

        print("\nConfiguration")
        print("-" * 40)
        print(f"Embedding model : {self.embedding_model_id}")
        print(f"Embedding dim   : {self.embedding_dimension}")
        print(f"Embedding batch : {self.embedding_batch_size}")
        print(f"Reranker model  : {self.reranker_model_id}")
        print(f"Reranker top-k  : {self.reranker_top_k}")
        print(f"LLM model       : {self.llm_model_id}")
        print(f"Qdrant          : {self.qdrant_collection_name}")

        print("\nHealth check passed.")

    def _check_environment(self) -> None:
        if not self.mineru_token:
            raise RuntimeError("MINERU_TOKEN is not configured.")

        if not self.google_api_key:
            raise RuntimeError("GOOGLE_API is not configured.")

    def _check_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mineru_output_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_output_dir.mkdir(parents=True, exist_ok=True)

        if not self.documents_dir.exists():
            raise RuntimeError(
                f"Documents directory does not exist: {self.documents_dir}"
            )

    def _check_dependencies(self) -> None:
        required = {
            "sentence_transformers": "sentence-transformers",
            "qdrant_client": "qdrant-client",
            "google.genai": "google-genai",
        }

        missing = []

        for module, package in required.items():
            try:
                __import__(module)
            except ImportError:
                missing.append(package)

        if missing:
            raise RuntimeError(
                "Missing required packages: " + ", ".join(missing)
            )


config = RAGConfig()
