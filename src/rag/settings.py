import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class RAGSettings(BaseSettings):

    google_api: str = os.environ["GOOGLE_API"]

    output_dir: Path = Path.cwd() / "outputs"
    qdrant_output_dir: Path = Path.cwd() / "outputs" / "qdrant"
    mineru_output_dir: Path = Path.cwd() / "outputs" /  "mineru"
    documents_dir : Path = Path.cwd() / "tests" / "documents"

    embedding_model_id: str = "bge-m3"
    reranker_model_id: str = "bge-reranker-v2-m3"
    llm_model_id : str = ""

    qdrant_collection_name: str = "knowledge_base"

    embedding_size: int = 1024

    model_config = SettingsConfigDict(env_prefix="RAG_")

settings = RAGSettings()
