from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from rag.main import logger
from rag.vlm import describe_image


IMAGE_PATTERN = re.compile(r"\[(.*?)\]\(.*?\)")
HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
splitter = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT_ON, strip_headers=False)


def create_chunks() -> list[Document]:
    final_chunks: list[Document] = []
    cwd = Path.cwd()

    logger.info(f"Searching for Markdown files in {cwd}")

    for md_file in cwd.glob("*/*/full.md"):
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


def refine_chunks(chunk: Document, file: Path) -> Document:
    # TODO: handle table too

    images: list[str] = IMAGE_PATTERN.findall(chunk.page_content)
    chunk.metadata.setdefault("images", [])
    chunk.metadata["source"] = str(file)
    logger.debug(f"Found {len(images)} images in chunk")

    for image in images:
        image_path = file.parent / image
        logger.debug(f"Image reference: {image}")
        logger.debug(f"Resolved image path: {image_path}")

        if not image_path.exists():
            raise RuntimeError(f"Cannot locate image: {image_path}")

        image_description = describe_image(image_path, chunk.page_content)
        logger.debug(f"Image description: {image_description}")
        image_reference = f"[{image}]({image})"
        chunk.page_content = chunk.page_content.replace(image_reference, image_description)
        chunk.metadata["images"].append(str(image_path))

    logger.debug(f"Refined chunk:\n{chunk.page_content}")
    logger.debug(f"Chunk metadata: {chunk.metadata}")

    return chunk


if __name__ == "__main__" :
    chunks = create_chunks()
    logger.info(f"{len(chunks)} chuncks created")

