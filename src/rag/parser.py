import requests

from pathlib import Path
from rag.main import logger
from contextlib import ExitStack
from rag.utils import chech_mineru_health
from rag.config import MINERU_BASE_URL, MINERU_PAYLOAD, SUPPORTED_EXTENSIONS


def parser(source: Path):
    source = source.expanduser().resolve()

    logger.info(f"Starting document parsing from {source}")

    if not source.exists():
        logger.error(f"Source does not exist: {source}")
        raise FileNotFoundError(source)

    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported document format: {source}")
            return 

        documents = [source]

    elif source.is_dir():
        documents = [
            file
            for file in source.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    else:
        logger.error(f"Source is neither a file nor a directory: {source}")
        raise ValueError(f"Invalid source: {source}")

    if not documents:
        logger.warning(f"No supported documents found in {source}")
        return 

    logger.info(f"Found {len(documents)} documents to parse")
    logger.debug(f"Documents: {', '.join(file.name for file in documents)}")

    with ExitStack() as stack:
        files = [
            ("files", (file.name, stack.enter_context(file.open("rb")), "application/octet-stream"))
            for file in documents
        ]

        logger.info(f"Sending {len(documents)} documents to MinerU at {MINERU_BASE_URL}")

        try:
            response = requests.post(
                f"{MINERU_BASE_URL}/file_parse",
                data=MINERU_PAYLOAD,
                files=files,
            )
            response.raise_for_status()
        except requests.RequestException: # TODO: I should fix this later
            logger.exception("MinerU parsing request failed")
            raise

    logger.info(f"MinerU successfully parsed {len(documents)} documents")
    logger.debug(f"MinerU response size: {response.content} bytes")


if __name__ == "__main__":
    logger.info("Starting MinerU parser")

    cwd = Path.cwd()

    chech_mineru_health()
    source = cwd/ "tests" / "documents"
    parser(source)
