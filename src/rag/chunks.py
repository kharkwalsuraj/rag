import logging
import re

from pathlib import Path

from html_table_parse import to_dicts
from rag.vlm import describe_image
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


logger = logging.getLogger(__name__)
IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")
TABLE_PATTERN = re.compile(r"")
HEADERS_TO_SPLIT_ON = [("#", "header1"), ("##", "header2"), ("###", "header3"), ("####", "header4")]
splitter = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT_ON, strip_headers=False)


def create_chunks() -> list[Document]:
    final_chunks: list[Document] = []
    cwd = Path.cwd()

    logger.info(f"Searching for Markdown files in {cwd}")

    for md_file in cwd.glob("output/*/*/*/*.md"):
        logger.info(f"Processing: {md_file}")

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        logger.debug(f"Loaded {len(content)} characters from {md_file}")
        chunks = splitter.split_text(content)
        logger.info(f"Created {len(chunks)} chunks from {md_file}")

        for chunk in chunks:
            final_chunks.append(refine_chunks(chunk, md_file))

    logger.info(f"Created {len(final_chunks)} final chunks")

    return final_chunks

def flatten_table(table_html: str) -> str:
    rows = to_dicts(table_html)

    return "\n".join(
        " | ".join(f"{key}: {value}" for key, value in row.items())
        for row in rows
    )


def refine_chunks(chunk: Document, file: Path) -> Document:
    chunk.metadata["source"] = str(file)
    chunk.metadata.setdefault("images", [])
    chunk.metadata.setdefault("tables_html", [])

    images: list[str] = IMAGE_PATTERN.findall(chunk.page_content)
    logger.debug(f"Found {len(images)} images in chunk: {images}")

    for image in images:
        image_path = file.parent / image
        logger.debug(f"Resolved image path: {image_path}")

        if not image_path.is_file():
            raise RuntimeError(f"Cannot locate image: {image_path}")

        image_description = describe_image(image_path, chunk.page_content)
        logger.debug(f"Image description: {image_description}")

        image_reference = f"![]({image})"
        chunk.page_content = chunk.page_content.replace(image_reference, image_description)
        chunk.metadata["images"].append(str(image_path))

    tables: list[str] = TABLE_PATTERN.findall(chunk.page_content)
    logger.debug(f"Found {len(tables)} tables in chunk")

    for table in tables:
        flattened_table = flatten_table(table)
        chunk.page_content = chunk.page_content.replace(table, flattened_table)
        chunk.metadata["tables_html"].append(table)
        logger.debug(f"Flattened table:\n{flattened_table}")

    logger.debug(f"Refined chunk:\n{chunk.page_content}")
    logger.debug(f"Chunk metadata: {chunk.metadata}")

    return chunk

if __name__ == "__main__" :
    chunks = create_chunks()
    print(chunks)
    logger.info(f"{len(chunks)} chuncks created")
